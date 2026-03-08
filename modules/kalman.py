import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import deeplabcut
import numpy as np
import pandas as pd
from tqdm import tqdm


def _normalize_videos(videos: str | Iterable[str]) -> list[Path]:
    return [Path(v).resolve() for v in deeplabcut.auxiliaryfunctions.get_list_of_videos(videos, "")]


def _frame_number_from_index(index_value: Any) -> int | None:
    if isinstance(index_value, tuple) and index_value:
        index_value = index_value[-1]
    token = str(index_value)
    match = re.search(r"img(\d+)", token, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)", token)
    return int(match.group(1)) if match else None


def _load_analyzed_predictions(
    config_path: str,
    video_path: Path,
    shuffle: int | None,
    destfolder: str | None,
) -> tuple[pd.DataFrame, Path]:
    output_dir = Path(destfolder).resolve() if destfolder else video_path.parent.resolve()
    vname = video_path.stem
    cfg = deeplabcut.auxiliaryfunctions.read_config(config_path)

    if shuffle is not None:
        for train_fraction in cfg.get("TrainingFraction", [0.95]):
            scorer, scorer_legacy = deeplabcut.auxiliaryfunctions.get_scorer_name(
                cfg,
                shuffle,
                trainFraction=train_fraction,
            )
            for s in (scorer, scorer_legacy):
                try:
                    df, filepath, _, _ = deeplabcut.auxiliaryfunctions.load_analyzed_data(
                        str(output_dir),
                        vname,
                        s,
                        filtered=False,
                    )
                    return df, Path(filepath)
                except FileNotFoundError:
                    continue

    candidates = sorted(output_dir.glob(f"{vname}*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    ignored_suffixes = (
        "_filtered.h5",
        "_meta.h5",
        "_full.h5",
        "_assemblies.h5",
        "_labeled.h5",
        "_skeleton.h5",
    )
    filtered_candidates = [
        p for p in candidates if not p.name.endswith(ignored_suffixes) and "_filtered" not in p.name
    ]
    if not filtered_candidates:
        raise FileNotFoundError(
            f"Could not find DLC prediction h5 for video '{video_path}' in '{output_dir}'."
        )
    latest_h5 = filtered_candidates[0]
    return pd.read_hdf(latest_h5), latest_h5


def _load_keypoints_from_df(df: pd.DataFrame, source_label: str = "") -> tuple[np.ndarray, list[str]]:
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(f"Unexpected DLC result format in {source_label or 'dataframe'}; expected MultiIndex columns.")

    scorer_values = set(map(str, df.columns.get_level_values(0)))
    has_single_scorer = len(scorer_values) == 1

    groups: dict[tuple, dict[str, tuple]] = {}
    ordered_keys: list[tuple] = []
    for col in df.columns.to_flat_index():
        coord = str(col[-1]).lower()
        if coord not in {"x", "y", "likelihood"}:
            continue
        key = tuple(col[:-1])
        if key not in groups:
            groups[key] = {}
            ordered_keys.append(key)
        groups[key][coord] = col

    valid_keys = [k for k in ordered_keys if "x" in groups[k] and "y" in groups[k]]
    n_frames = len(df)
    n_joints = len(valid_keys)
    keypoints = np.full((n_frames, n_joints, 3), np.nan, dtype=np.float64)
    joint_names: list[str] = []

    for j, key in enumerate(valid_keys):
        col_map = groups[key]
        keypoints[:, j, 0] = pd.to_numeric(df[col_map["x"]], errors="coerce").to_numpy()
        keypoints[:, j, 1] = pd.to_numeric(df[col_map["y"]], errors="coerce").to_numpy()
        if "likelihood" in col_map:
            keypoints[:, j, 2] = pd.to_numeric(
                df[col_map["likelihood"]],
                errors="coerce",
            ).to_numpy()
        else:
            keypoints[:, j, 2] = 1.0

        name_parts = [str(p) for p in key if str(p).lower() != "nan"]
        if has_single_scorer and name_parts:
            name_parts = name_parts[1:]
        joint_names.append("|".join(name_parts) if name_parts else f"joint_{j}")

    return keypoints, joint_names


def _build_skeleton_edges(config_path: str, joint_names: list[str]) -> list[tuple[int, int]]:
    cfg = deeplabcut.auxiliaryfunctions.read_config(config_path)
    skeleton = cfg.get("skeleton", []) or []
    if not skeleton:
        return []

    parsed: list[tuple[str, str]] = []
    for name in joint_names:
        parts = name.split("|")
        if len(parts) >= 2:
            parsed.append((parts[-2], parts[-1]))
        else:
            parsed.append(("", parts[-1]))

    individuals = sorted(set(ind for ind, _ in parsed))
    index_map = {(ind, bp): i for i, (ind, bp) in enumerate(parsed)}
    edges: list[tuple[int, int]] = []

    for pair in skeleton:
        if len(pair) != 2:
            continue
        bp_a, bp_b = str(pair[0]), str(pair[1])
        for ind in individuals:
            ia = index_map.get((ind, bp_a))
            ib = index_map.get((ind, bp_b))
            if ia is not None and ib is not None:
                edges.append((ia, ib))
    return edges


def _estimate_scale_from_keypoints(
    keypoints: np.ndarray,
    edges: list[tuple[int, int]],
    min_valid: int = 10,
) -> float:
    lengths: list[np.ndarray] = []
    if edges:
        for i, j in edges:
            pi = keypoints[:, i, :2]
            pj = keypoints[:, j, :2]
            valid = np.isfinite(pi).all(axis=1) & np.isfinite(pj).all(axis=1)
            if np.count_nonzero(valid) == 0:
                continue
            lengths.append(np.linalg.norm(pi[valid] - pj[valid], axis=1))
    if lengths:
        stacked = np.concatenate(lengths)
        if stacked.size >= min_valid:
            return float(np.median(stacked))

    finite = np.isfinite(keypoints[:, :, :2]).all(axis=2)
    diag_vals: list[float] = []
    for t in range(keypoints.shape[0]):
        valid_idx = np.where(finite[t])[0]
        if valid_idx.size < 2:
            continue
        pts = keypoints[t, valid_idx, :2]
        span = np.max(pts, axis=0) - np.min(pts, axis=0)
        diag = float(np.linalg.norm(span))
        if np.isfinite(diag) and diag > 0:
            diag_vals.append(diag)
    if diag_vals:
        return float(np.median(diag_vals))
    return 200.0


def _joint_key_variants(joint_name: str) -> list[str]:
    parts = joint_name.split("|")
    variants = [joint_name]
    if parts:
        variants.append(parts[-1])
    if len(parts) >= 2:
        variants.append("|".join(parts[-2:]))
    dedup: list[str] = []
    for item in variants:
        if item and item not in dedup:
            dedup.append(item)
    return dedup


class JointKalmanFilter:
    def __init__(
        self,
        dt: float = 1.0,
        process_var: float = 1.0,
        obs_var: float = 12.0,
        gate_threshold: float = 80.0,
        velocity_damping: float = 0.8,
        hard_reject_scale: float = 3.0,
        max_missed_updates: int = 5,
        reinit_confidence: float = 0.6,
        min_confidence: float = 1e-3,
    ):
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1e3
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, velocity_damping, 0],
                [0, 0, 0, velocity_damping],
            ],
            dtype=np.float64,
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.Q = np.eye(4, dtype=np.float64) * process_var
        self.obs_var = obs_var
        self.gate_threshold = gate_threshold
        self.hard_reject_threshold = gate_threshold * hard_reject_scale
        self.max_missed_updates = max_missed_updates
        self.reinit_confidence = reinit_confidence
        self.min_confidence = min_confidence
        self.init_covariance = np.diag([50.0, 50.0, 100.0, 100.0]).astype(np.float64)
        self.missed_updates = 0
        self.last_observation_conf = 0.0
        self.initialized = False

    def _initialize(self, z: np.ndarray, conf: float) -> None:
        self.x[:2] = z
        self.x[2:] = 0.0
        self.P = self.init_covariance.copy()
        self.initialized = True
        self.missed_updates = 0
        self.last_observation_conf = conf

    @staticmethod
    def _residual_scale(dist: float, gate: float) -> float:
        if dist <= gate:
            return 1.0
        if dist <= 2.0 * gate:
            return 5.0
        return 20.0

    def step(
        self,
        measurement: np.ndarray | None,
        confidence: float | None,
        update_conf_min: float = 0.2,
    ) -> tuple[np.ndarray, dict[str, float | bool]]:
        if self.initialized:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q

        used_measurement = False
        conf = (
            float(np.clip(confidence, self.min_confidence, 1.0))
            if confidence is not None and np.isfinite(confidence)
            else self.min_confidence
        )
        has_valid_measurement = (
            measurement is not None
            and np.all(np.isfinite(measurement))
            and confidence is not None
            and np.isfinite(confidence)
            and float(confidence) >= update_conf_min
        )
        innovation_dist = float("inf")
        residual_scale = float("inf")
        weak_update = False

        if has_valid_measurement:
            z = measurement.reshape(2, 1).astype(np.float64)

            if not self.initialized:
                self._initialize(z, conf)
                used_measurement = True
                innovation_dist = 0.0
                residual_scale = 1.0
            else:
                predicted_xy = self.H @ self.x
                dist = float(np.linalg.norm((z - predicted_xy).ravel()))
                innovation_dist = dist

                if dist > self.hard_reject_threshold:
                    self.missed_updates += 1
                    if self.missed_updates >= self.max_missed_updates and conf >= self.reinit_confidence:
                        self._initialize(z, conf)
                        used_measurement = True
                        innovation_dist = 0.0
                        residual_scale = 1.0
                else:
                    residual_scale = self._residual_scale(dist, self.gate_threshold)
                    weak_update = residual_scale > 5.0
                    r_scale = (1.0 / (conf * conf)) * residual_scale
                    R = np.eye(2, dtype=np.float64) * (self.obs_var * r_scale)
                    y = z - self.H @ self.x
                    S = self.H @ self.P @ self.H.T + R
                    K = self.P @ self.H.T @ np.linalg.inv(S)
                    self.x = self.x + K @ y
                    self.P = (np.eye(4, dtype=np.float64) - K @ self.H) @ self.P
                    if weak_update:
                        self.missed_updates += 1
                    else:
                        self.missed_updates = 0
                    self.last_observation_conf = float(np.clip(conf / residual_scale, 0.0, 1.0))
                    used_measurement = True
        elif self.initialized:
            self.missed_updates += 1

        if not self.initialized:
            return np.array([np.nan, np.nan], dtype=np.float64), {
                "used_measurement": False,
                "frames_since_update": float("inf"),
                "pos_std": float("inf"),
                "effective_conf": 0.0,
                "measurement_conf": 0.0,
                "innovation_dist": float("inf"),
                "residual_scale": float("inf"),
                "weak_update": False,
            }

        pos_std = float(np.sqrt(np.trace(self.P[:2, :2])))
        effective_conf = float(
            np.clip(self.last_observation_conf * np.exp(-0.7 * self.missed_updates), 0.0, 1.0)
        )

        return self.x[:2, 0].copy(), {
            "used_measurement": used_measurement,
            "frames_since_update": float(self.missed_updates),
            "pos_std": pos_std,
            "effective_conf": effective_conf,
            "measurement_conf": conf if has_valid_measurement else 0.0,
            "innovation_dist": innovation_dist,
            "residual_scale": residual_scale,
            "weak_update": weak_update,
        }


class SkeletonKalmanTracker:
    def __init__(
        self,
        n_joints: int,
        edges: list[tuple[int, int]] | None = None,
        process_var: float = 1.0,
        obs_var: float = 12.0,
        gate_threshold: float = 80.0,
        velocity_damping: float = 0.8,
        max_missed_updates: int = 5,
        reinit_confidence: float = 0.6,
        bone_length_tol: float = 0.35,
        bone_adjust_alpha: float = 0.6,
        bone_confidence_anchor: float = 0.4,
        update_conf_min: float = 0.2,
        render_conf_min: float = 0.25,
        max_extrapolation_frames: int = 2,
        max_pos_std: float = 30.0,
        per_joint_params: list[dict[str, Any]] | None = None,
    ):
        self.edges = edges or []
        self.bone_length_tol = bone_length_tol
        self.bone_adjust_alpha = bone_adjust_alpha
        self.bone_confidence_anchor = bone_confidence_anchor
        self.update_conf_min_arr = np.full(n_joints, update_conf_min, dtype=np.float64)
        self.render_conf_min_arr = np.full(n_joints, render_conf_min, dtype=np.float64)
        self.max_extrapolation_arr = np.full(n_joints, max_extrapolation_frames, dtype=np.int32)
        self.max_pos_std_arr = np.full(n_joints, max_pos_std, dtype=np.float64)

        self.filters: list[JointKalmanFilter] = []
        for j in range(n_joints):
            joint_cfg = per_joint_params[j] if per_joint_params and j < len(per_joint_params) else {}
            joint_process_var = float(joint_cfg.get("process_var", process_var))
            joint_obs_var = float(joint_cfg.get("obs_var", obs_var))
            joint_gate = float(joint_cfg.get("gate_threshold", gate_threshold))
            joint_damping = float(joint_cfg.get("velocity_damping", velocity_damping))
            joint_max_missed = int(joint_cfg.get("max_missed_updates", max_missed_updates))
            joint_reinit = float(joint_cfg.get("reinit_confidence", reinit_confidence))

            self.filters.append(
                JointKalmanFilter(
                    process_var=joint_process_var,
                    obs_var=joint_obs_var,
                    gate_threshold=joint_gate,
                    velocity_damping=joint_damping,
                    max_missed_updates=joint_max_missed,
                    reinit_confidence=joint_reinit,
                )
            )

            self.update_conf_min_arr[j] = float(joint_cfg.get("update_conf_min", self.update_conf_min_arr[j]))
            self.render_conf_min_arr[j] = float(joint_cfg.get("render_conf_min", self.render_conf_min_arr[j]))
            self.max_extrapolation_arr[j] = int(
                joint_cfg.get("max_extrapolation_frames", self.max_extrapolation_arr[j])
            )
            self.max_pos_std_arr[j] = float(joint_cfg.get("max_pos_std", self.max_pos_std_arr[j]))

    def _estimate_reference_bone_lengths(self, keypoints: np.ndarray) -> dict[tuple[int, int], float]:
        ref_lengths: dict[tuple[int, int], float] = {}
        if not self.edges:
            return ref_lengths
        for i, j in self.edges:
            pi = keypoints[:, i, :2]
            pj = keypoints[:, j, :2]
            ci = keypoints[:, i, 2]
            cj = keypoints[:, j, 2]
            valid = (
                np.isfinite(pi).all(axis=1)
                & np.isfinite(pj).all(axis=1)
                & (ci >= self.bone_confidence_anchor)
                & (cj >= self.bone_confidence_anchor)
            )
            if not np.any(valid):
                continue
            lengths = np.linalg.norm(pi[valid] - pj[valid], axis=1)
            if lengths.size:
                ref_lengths[(i, j)] = float(np.median(lengths))
        return ref_lengths

    def _suppress_implausible_bones(
        self,
        frame_points: np.ndarray,
        frame_visible: np.ndarray,
        frame_effective_conf: np.ndarray,
        ref_lengths: dict[tuple[int, int], float],
    ) -> None:
        for i, j in self.edges:
            ref_len = ref_lengths.get((i, j))
            if ref_len is None or ref_len <= 0 or not (frame_visible[i] and frame_visible[j]):
                continue

            xi, yi = frame_points[i, :2]
            xj, yj = frame_points[j, :2]
            if not np.isfinite([xi, yi, xj, yj]).all():
                continue

            dist = float(np.linalg.norm(np.array([xj - xi, yj - yi], dtype=np.float64)))
            rel_err = abs(dist - ref_len) / ref_len
            hide_tol = max(0.75, self.bone_length_tol * 2.0)
            if rel_err <= hide_tol:
                continue

            ci = frame_effective_conf[i]
            cj = frame_effective_conf[j]
            render_floor = float(np.median(self.render_conf_min_arr))
            if max(ci, cj) < render_floor + 0.1 and abs(ci - cj) < 0.25:
                continue
            if ci >= cj:
                frame_visible[j] = False
            else:
                frame_visible[i] = False

    def _apply_bone_constraints(
        self,
        frame_points: np.ndarray,
        ref_lengths: dict[tuple[int, int], float],
    ) -> None:
        for i, j in self.edges:
            ref_len = ref_lengths.get((i, j))
            if ref_len is None or ref_len <= 0:
                continue

            xi, yi, ci = frame_points[i]
            xj, yj, cj = frame_points[j]
            if not np.isfinite([xi, yi, xj, yj]).all():
                continue

            vec = np.array([xj - xi, yj - yi], dtype=np.float64)
            dist = float(np.linalg.norm(vec))
            if dist < 1e-6:
                continue

            rel_err = abs(dist - ref_len) / ref_len
            if rel_err <= self.bone_length_tol:
                continue

            if ci >= cj and ci >= self.bone_confidence_anchor:
                anchor = np.array([xi, yi], dtype=np.float64)
                moving = np.array([xj, yj], dtype=np.float64)
                move_index = j
            elif cj > ci and cj >= self.bone_confidence_anchor:
                anchor = np.array([xj, yj], dtype=np.float64)
                moving = np.array([xi, yi], dtype=np.float64)
                move_index = i
            else:
                continue

            direction = (moving - anchor) / dist
            target = anchor + direction * ref_len
            corrected = moving * (1.0 - self.bone_adjust_alpha) + target * self.bone_adjust_alpha
            frame_points[move_index, 0] = corrected[0]
            frame_points[move_index, 1] = corrected[1]

    def smooth(self, keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        smoothed = np.array(keypoints, copy=True)
        n_frames, n_joints, _ = keypoints.shape
        visible = np.zeros((n_frames, n_joints), dtype=bool)
        effective_conf = np.zeros((n_frames, n_joints), dtype=np.float64)
        ref_lengths = self._estimate_reference_bone_lengths(keypoints)
        for t in range(n_frames):
            for j in range(n_joints):
                meas = keypoints[t, j, :2]
                conf = keypoints[t, j, 2] if keypoints.shape[2] > 2 else 1.0
                if not np.all(np.isfinite(meas)):
                    meas_in = None
                else:
                    meas_in = meas
                xy, state = self.filters[j].step(
                    meas_in,
                    conf,
                    update_conf_min=float(self.update_conf_min_arr[j]),
                )
                smoothed[t, j, 0] = xy[0]
                smoothed[t, j, 1] = xy[1]
                effective_conf[t, j] = float(state["effective_conf"])
                smoothed[t, j, 2] = effective_conf[t, j]
                joint_render_min = float(self.render_conf_min_arr[j])
                joint_max_pos_std = float(self.max_pos_std_arr[j])
                joint_max_extra = int(self.max_extrapolation_arr[j])
                joint_gate = float(self.filters[j].gate_threshold)
                if bool(state["used_measurement"]):
                    visible[t, j] = bool(
                        np.isfinite(xy).all()
                        and state["measurement_conf"] >= joint_render_min
                        and state["innovation_dist"] <= (2.5 * joint_gate)
                        and state["pos_std"] <= (1.6 * joint_max_pos_std)
                        and (not bool(state["weak_update"]) or state["measurement_conf"] >= 0.75)
                    )
                else:
                    visible[t, j] = bool(
                        np.isfinite(xy).all()
                        and state["frames_since_update"] <= joint_max_extra
                        and state["pos_std"] <= joint_max_pos_std
                        and state["effective_conf"] >= joint_render_min
                    )
            if ref_lengths:
                self._apply_bone_constraints(smoothed[t], ref_lengths)
                self._suppress_implausible_bones(smoothed[t], visible[t], effective_conf[t], ref_lengths)
        return smoothed, visible, effective_conf


def _draw_pose(
    frame: np.ndarray,
    points: np.ndarray,
    edges: list[tuple[int, int]],
    visible_mask: np.ndarray,
    effective_conf: np.ndarray,
    conf_thresh: float = 0.1,
) -> np.ndarray:
    out = frame
    for i, j in edges:
        xi, yi, _ = points[i]
        xj, yj, _ = points[j]
        if (
            visible_mask[i]
            and visible_mask[j]
            and effective_conf[i] >= conf_thresh
            and effective_conf[j] >= conf_thresh
            and np.isfinite([xi, yi, xj, yj]).all()
        ):
            cv2.line(out, (int(round(xi)), int(round(yi))), (int(round(xj)), int(round(yj))), (0, 255, 255), 2)

    for idx, (x, y, _) in enumerate(points):
        if visible_mask[idx] and effective_conf[idx] >= conf_thresh and np.isfinite([x, y]).all():
            cv2.circle(out, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
    return out


def _resolve_runtime_kalman_config(
    kalman_params: dict[str, Any] | None,
    joint_names: list[str],
    keypoints: np.ndarray,
    edges: list[tuple[int, int]],
    defaults: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime = dict(defaults)
    per_joint_cfg = [{} for _ in joint_names]
    if not kalman_params:
        return runtime, per_joint_cfg

    video_scale = _estimate_scale_from_keypoints(keypoints, edges)
    global_cfg = kalman_params.get("global", {})
    if isinstance(global_cfg, dict):
        runtime.update({k: v for k, v in global_cfg.items() if k in runtime})
        if "gate_threshold_norm" in global_cfg and "gate_threshold" not in global_cfg:
            runtime["gate_threshold"] = float(global_cfg["gate_threshold_norm"]) * video_scale
        if "max_pos_std_norm" in global_cfg and "max_pos_std" not in global_cfg:
            runtime["max_pos_std"] = float(global_cfg["max_pos_std_norm"]) * video_scale

    per_joint_map = kalman_params.get("per_joint", {})
    if not isinstance(per_joint_map, dict):
        return runtime, per_joint_cfg

    for j, name in enumerate(joint_names):
        cfg: dict[str, Any] = {}
        for key in _joint_key_variants(name):
            src = per_joint_map.get(key)
            if isinstance(src, dict):
                cfg.update(src)
        if "gate_threshold_norm" in cfg and "gate_threshold" not in cfg:
            cfg["gate_threshold"] = float(cfg["gate_threshold_norm"]) * video_scale
        if "max_pos_std_norm" in cfg and "max_pos_std" not in cfg:
            cfg["max_pos_std"] = float(cfg["max_pos_std_norm"]) * video_scale
        per_joint_cfg[j] = cfg

    return runtime, per_joint_cfg


def _find_collected_h5(label_dir: Path, scorer: str) -> Path | None:
    exact = label_dir / f"CollectedData_{scorer}.h5"
    if exact.exists():
        return exact
    candidates = sorted(label_dir.glob("CollectedData_*.h5"))
    return candidates[0] if candidates else None


def _subsample_indices(n_items: int, max_items: int) -> np.ndarray:
    if n_items <= max_items:
        return np.arange(n_items, dtype=np.int64)
    step = int(np.ceil(n_items / max_items))
    idx = np.arange(0, n_items, step, dtype=np.int64)
    if idx.size > max_items:
        idx = idx[:max_items]
    return idx


def _append_capped_samples(dst: list[float], samples: np.ndarray, cap: int) -> None:
    if samples.size == 0 or len(dst) >= cap:
        return
    remain = cap - len(dst)
    if samples.size > remain:
        take_idx = _subsample_indices(samples.size, remain)
        samples = samples[take_idx]
    dst.extend(samples.astype(np.float64).tolist())


def _index_to_stem_image_key(index_value: Any) -> str:
    if isinstance(index_value, tuple):
        parts = [str(v) for v in index_value if str(v).lower() != "nan"]
        if len(parts) >= 2:
            stem = Path(parts[-2]).stem
            img = Path(parts[-1]).name
            return f"{stem}/{img}".lower()
        index_value = parts[-1] if parts else ""
    text = str(index_value)
    path = Path(text)
    stem = path.parent.name
    img = path.name
    if stem:
        return f"{stem}/{img}".lower()
    return img.lower()


def _run_model_predictions_on_images(
    config_path: str,
    image_paths: list[Path],
    shuffle: int,
    trainingsetindex: int,
    modelprefix: str,
    cache_dir: Path,
    use_cached_predictions: bool = True,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "kalman_fit_image_predictions.h5"
    if use_cached_predictions and cache_file.exists():
        return pd.read_hdf(cache_file)

    for old_h5 in cache_dir.glob("*.h5"):
        old_h5.unlink(missing_ok=True)

    deeplabcut.analyze_images(
        config=config_path,
        images=[str(p) for p in image_paths],
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        save_as_csv=False,
        destfolder=str(cache_dir),
        modelprefix=modelprefix,
    )

    produced_h5 = sorted(cache_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not produced_h5:
        raise FileNotFoundError(f"No image prediction h5 produced in {cache_dir}")
    pred_df = pd.read_hdf(produced_h5[0])
    pred_df.to_hdf(cache_file, key="predictions", mode="w")
    return pred_df


def _fit_joint_params_from_stats(
    speed_samples: np.ndarray,
    missing_ratio: float,
    visibility_labels: np.ndarray | None = None,
    confidence_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    if speed_samples.size == 0:
        q50 = 0.006
        q90 = 0.012
        q95 = 0.018
        sstd = 0.004
    else:
        q50 = float(np.quantile(speed_samples, 0.50))
        q90 = float(np.quantile(speed_samples, 0.90))
        q95 = float(np.quantile(speed_samples, 0.95))
        sstd = float(np.std(speed_samples))

    gate_norm = float(np.clip(0.02 + 8.0 * q95 + 2.0 * q50, 0.04, 0.45))
    max_pos_std_norm = float(np.clip(0.08 + 2.2 * gate_norm, 0.12, 0.9))
    velocity_damping = float(np.clip(0.92 - 5.0 * sstd, 0.65, 0.92))
    process_var = float(np.clip(0.4 + 40.0 * q90, 0.2, 8.0))
    obs_var = float(np.clip(6.0 + 220.0 * q50, 4.0, 40.0))
    max_extrapolation_frames = 1 if missing_ratio > 0.60 else (2 if missing_ratio > 0.35 else 3)

    render_conf_min = float(np.clip(0.18 + 0.40 * missing_ratio, 0.18, 0.75))
    if visibility_labels is not None and confidence_scores is not None:
        labels = visibility_labels.astype(np.int32)
        confs = confidence_scores.astype(np.float64)
        valid = np.isfinite(confs)
        labels = labels[valid]
        confs = confs[valid]
        if labels.size >= 20 and np.unique(labels).size > 1:
            cand = np.unique(np.concatenate([np.linspace(0.05, 0.95, 37), np.clip(confs, 0.0, 1.0)]))
            best_t = render_conf_min
            best_score = -1.0
            beta2 = 0.25  # F0.5: precision-biased, less false positive display for occluded points.
            for t in cand:
                pred = confs >= t
                tp = float(np.sum(pred & (labels == 1)))
                fp = float(np.sum(pred & (labels == 0)))
                fn = float(np.sum((~pred) & (labels == 1)))
                denom = (1 + beta2) * tp + beta2 * fn + fp
                score = ((1 + beta2) * tp / denom) if denom > 0 else 0.0
                if score > best_score or (abs(score - best_score) < 1e-9 and t > best_t):
                    best_score = score
                    best_t = float(t)
            render_conf_min = float(np.clip(best_t, 0.15, 0.90))

    update_conf_min = float(np.clip(render_conf_min - 0.08, 0.08, 0.70))

    return {
        "gate_threshold_norm": gate_norm,
        "max_pos_std_norm": max_pos_std_norm,
        "render_conf_min": render_conf_min,
        "update_conf_min": update_conf_min,
        "max_extrapolation_frames": int(max_extrapolation_frames),
        "velocity_damping": velocity_damping,
        "process_var": process_var,
        "obs_var": obs_var,
    }


def kalman_fitter(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
) -> dict[str, Any]:
    """Fit scale-aware Kalman parameters from labels + model predictions."""
    project_root = Path(project_path).resolve()
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found under {project_root}")

    cfg = deeplabcut.auxiliaryfunctions.read_config(str(config_path))
    video_sets = cfg.get("video_sets", {})
    if not video_sets:
        raise ValueError(f"No video_sets found in {config_path}")

    scorer = str(cfg.get("scorer", "")).strip()
    labeled_root = project_root / "labeled-data"
    if not labeled_root.exists():
        raise FileNotFoundError(f"labeled-data folder not found: {labeled_root}")

    cache_dir = project_root / ".kalman_fit_cache" / f"shuffle{shuffle}"
    video_scales: list[float] = []
    speed_by_joint: dict[str, list[float]] = defaultdict(list)
    vis_labels_by_joint: dict[str, list[int]] = defaultdict(list)
    conf_scores_by_joint: dict[str, list[float]] = defaultdict(list)
    missing_by_joint: dict[str, int] = defaultdict(int)
    total_by_joint: dict[str, int] = defaultdict(int)
    videos_used: list[str] = []
    sampled_records: list[dict[str, Any]] = []
    all_image_paths: list[Path] = []

    for video_path_str in video_sets.keys():
        stem = Path(video_path_str).stem
        label_dir = labeled_root / stem
        if not label_dir.exists():
            continue

        h5_path = _find_collected_h5(label_dir, scorer)
        if h5_path is None:
            continue

        df = pd.read_hdf(h5_path)
        keypoints, joint_names = _load_keypoints_from_df(df, source_label=str(h5_path))

        frame_numbers = np.array([_frame_number_from_index(i) for i in df.index], dtype=np.float64)
        valid_frame_idx = np.where(np.isfinite(frame_numbers))[0]
        if valid_frame_idx.size == 0:
            continue

        frame_numbers = frame_numbers[valid_frame_idx].astype(np.int64)
        keypoints = keypoints[valid_frame_idx]

        order = np.argsort(frame_numbers)
        frame_numbers = frame_numbers[order]
        keypoints = keypoints[order]

        sample_idx = _subsample_indices(keypoints.shape[0], max_labeled_frames_per_video)
        frame_numbers = frame_numbers[sample_idx]
        keypoints = keypoints[sample_idx]
        sampled_index = np.array(df.index, dtype=object)[valid_frame_idx][order][sample_idx]

        image_paths: list[Path] = []
        image_keys: list[str] = []
        for idx_item in sampled_index:
            img_name = Path(str(idx_item[-1] if isinstance(idx_item, tuple) else idx_item)).name
            img_path = (label_dir / img_name).resolve()
            if not img_path.exists():
                continue
            image_paths.append(img_path)
            image_keys.append(f"{stem}/{img_path.name}".lower())
        if not image_paths:
            continue

        edges = _build_skeleton_edges(str(config_path), joint_names)
        video_scale = _estimate_scale_from_keypoints(keypoints, edges)
        if np.isfinite(video_scale) and video_scale > 1e-6:
            video_scales.append(float(video_scale))
        sampled_records.append(
            {
                "stem": stem,
                "joint_names": joint_names,
                "edges": edges,
                "scale": float(video_scale),
                "frame_numbers": frame_numbers[: len(image_paths)],
                "gt_xy": keypoints[: len(image_paths), :, :2],
                "image_keys": image_keys,
            }
        )
        all_image_paths.extend(image_paths)
        videos_used.append(stem)

    if not sampled_records:
        raise ValueError(
            f"No usable labeled frames found under {labeled_root}. "
            "Expected folders matching stems from config.yaml video_sets."
        )

    pred_df = _run_model_predictions_on_images(
        config_path=str(config_path),
        image_paths=all_image_paths,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        cache_dir=cache_dir,
        use_cached_predictions=use_cached_predictions,
    )
    pred_keypoints, pred_joint_names = _load_keypoints_from_df(
        pred_df,
        source_label=str(cache_dir / "kalman_fit_image_predictions.h5"),
    )
    pred_joint_idx = {name.split("|")[-1]: i for i, name in enumerate(pred_joint_names)}
    pred_lookup = {
        _index_to_stem_image_key(idx): pred_keypoints[i]
        for i, idx in enumerate(pred_df.index)
    }

    for rec in sampled_records:
        joint_names = rec["joint_names"]
        frame_numbers = np.asarray(rec["frame_numbers"], dtype=np.int64)
        gt_xy = rec["gt_xy"]
        image_keys = rec["image_keys"]
        video_scale = float(rec["scale"])

        pred_for_video = np.full((len(image_keys), len(joint_names), 3), np.nan, dtype=np.float64)
        valid_rows = np.zeros(len(image_keys), dtype=bool)
        for ridx, key in enumerate(image_keys):
            pred_row = pred_lookup.get(key)
            if pred_row is None:
                continue
            valid_rows[ridx] = True
            for j, joint_name in enumerate(joint_names):
                pidx = pred_joint_idx.get(joint_name.split("|")[-1])
                if pidx is not None:
                    pred_for_video[ridx, j] = pred_row[pidx]

        if not np.any(valid_rows):
            continue

        for j, joint_name in enumerate(joint_names):
            joint_key = joint_name.split("|")[-1]
            gt_joint = gt_xy[:, j]
            gt_visible = np.isfinite(gt_joint).all(axis=1)
            pred_joint = pred_for_video[:, j]
            pred_visible = np.isfinite(pred_joint[:, :2]).all(axis=1)
            pred_conf = pred_joint[:, 2]

            total_by_joint[joint_key] += int(gt_visible.size)
            missing_by_joint[joint_key] += int(gt_visible.size - np.count_nonzero(gt_visible))

            cls_mask = valid_rows & np.isfinite(pred_conf)
            if np.any(cls_mask):
                vis_labels_by_joint[joint_key].extend(gt_visible[cls_mask].astype(np.int32).tolist())
                conf_scores_by_joint[joint_key].extend(pred_conf[cls_mask].astype(np.float64).tolist())

            motion_mask = gt_visible & pred_visible & valid_rows
            valid_idx = np.where(motion_mask)[0]
            if valid_idx.size < 2:
                continue
            f = frame_numbers[valid_idx].astype(np.float64)
            c = pred_joint[valid_idx, :2]
            dt = np.diff(f)
            delta = np.linalg.norm(np.diff(c, axis=0), axis=1)
            good = dt > 0
            if not np.any(good):
                continue
            speed_norm = (delta[good] / dt[good]) / max(video_scale, 1e-6)
            _append_capped_samples(
                speed_by_joint[joint_key],
                speed_norm,
                max_speed_samples_per_joint,
            )

    if not total_by_joint:
        raise ValueError(
            "No matched prediction/label pairs were found for fitting. "
            "Please check shuffle and whether analyze_images can run on labeled frames."
        )

    bodyparts = [str(bp) for bp in cfg.get("bodyparts", [])]
    fitted_joint_names = bodyparts if bodyparts else sorted(total_by_joint.keys())

    all_speed = (
        np.concatenate([np.asarray(v, dtype=np.float64) for v in speed_by_joint.values() if len(v)], axis=0)
        if any(len(v) for v in speed_by_joint.values())
        else np.array([], dtype=np.float64)
    )
    all_labels = (
        np.concatenate([np.asarray(v, dtype=np.int32) for v in vis_labels_by_joint.values() if len(v)], axis=0)
        if any(len(v) for v in vis_labels_by_joint.values())
        else np.array([], dtype=np.int32)
    )
    all_confs = (
        np.concatenate([np.asarray(v, dtype=np.float64) for v in conf_scores_by_joint.values() if len(v)], axis=0)
        if any(len(v) for v in conf_scores_by_joint.values())
        else np.array([], dtype=np.float64)
    )
    global_missing_ratio = (
        float(sum(missing_by_joint.values()) / max(1, sum(total_by_joint.values())))
        if total_by_joint
        else 0.3
    )
    global_params = _fit_joint_params_from_stats(
        speed_samples=all_speed,
        missing_ratio=global_missing_ratio,
        visibility_labels=all_labels if all_labels.size else None,
        confidence_scores=all_confs if all_confs.size else None,
    )

    per_joint: dict[str, dict[str, Any]] = {}
    for joint_key in fitted_joint_names:
        speeds = np.asarray(speed_by_joint.get(joint_key, []), dtype=np.float64)
        if speeds.size == 0 and all_speed.size:
            speeds = all_speed
        miss_ratio = float(
            missing_by_joint.get(joint_key, 0) / max(1, total_by_joint.get(joint_key, 0))
        )
        if total_by_joint.get(joint_key, 0) == 0:
            miss_ratio = global_missing_ratio
        labels = np.asarray(vis_labels_by_joint.get(joint_key, []), dtype=np.int32)
        confs = np.asarray(conf_scores_by_joint.get(joint_key, []), dtype=np.float64)
        if labels.size == 0 and all_labels.size:
            labels = all_labels
            confs = all_confs
        per_joint[joint_key] = _fit_joint_params_from_stats(
            speed_samples=speeds,
            missing_ratio=miss_ratio,
            visibility_labels=labels if labels.size else None,
            confidence_scores=confs if confs.size else None,
        )

    global_defaults = {
        "process_var": global_params["process_var"],
        "obs_var": global_params["obs_var"],
        "gate_threshold_norm": global_params["gate_threshold_norm"],
        "velocity_damping": global_params["velocity_damping"],
        "max_missed_updates": 5,
        "reinit_confidence": 0.6,
        "bone_length_tol": 0.35,
        "bone_adjust_alpha": 0.6,
        "bone_confidence_anchor": 0.4,
        "update_conf_min": global_params["update_conf_min"],
        "render_conf_min": global_params["render_conf_min"],
        "max_extrapolation_frames": global_params["max_extrapolation_frames"],
        "max_pos_std_norm": global_params["max_pos_std_norm"],
        "draw_conf_thresh": 0.1,
    }

    return {
        "version": 2,
        "project_path": str(project_root),
        "shuffle": int(shuffle),
        "trainingsetindex": int(trainingsetindex),
        "videos_used": sorted(set(videos_used)),
        "scale_reference": float(np.median(video_scales)) if video_scales else 200.0,
        "global": global_defaults,
        "per_joint": per_joint,
    }


def kalman_video(
    config_path: str,
    videofile_path: str | Iterable[str],
    shuffle: int | None = None,
    destfolder: str | None = None,
    kalman_params: dict[str, Any] | None = None,
    process_var: float = 1.0,
    obs_var: float = 12.0,
    gate_threshold: float = 80.0,
    velocity_damping: float = 0.8,
    max_missed_updates: int = 5,
    reinit_confidence: float = 0.6,
    bone_length_tol: float = 0.35,
    bone_adjust_alpha: float = 0.6,
    bone_confidence_anchor: float = 0.4,
    update_conf_min: float = 0.2,
    render_conf_min: float = 0.25,
    max_extrapolation_frames: int = 2,
    max_pos_std: float = 30.0,
    draw_conf_thresh: float = 0.1,
    render_conf_min_offset: float = 0.0,
    update_conf_min_offset: float = 0.0,
    gate_threshold_offset: float = 0.0,
    max_pos_std_offset: float = 0.0,
    draw_conf_thresh_offset: float = 0.0,
) -> list[str]:
    """Render a video with Kalman-smoothed DLC keypoints."""
    videos = _normalize_videos(videofile_path)
    output_videos: list[str] = []

    for video_path in videos:
        pred_df, result_h5 = _load_analyzed_predictions(config_path, video_path, shuffle, destfolder)
        keypoints, joint_names = _load_keypoints_from_df(pred_df, source_label=str(result_h5))
        edges = _build_skeleton_edges(config_path, joint_names)
        runtime_defaults = {
            "process_var": process_var,
            "obs_var": obs_var,
            "gate_threshold": gate_threshold,
            "velocity_damping": velocity_damping,
            "max_missed_updates": max_missed_updates,
            "reinit_confidence": reinit_confidence,
            "bone_length_tol": bone_length_tol,
            "bone_adjust_alpha": bone_adjust_alpha,
            "bone_confidence_anchor": bone_confidence_anchor,
            "update_conf_min": update_conf_min,
            "render_conf_min": render_conf_min,
            "max_extrapolation_frames": max_extrapolation_frames,
            "max_pos_std": max_pos_std,
            "draw_conf_thresh": draw_conf_thresh,
        }
        runtime_cfg, per_joint_cfg = _resolve_runtime_kalman_config(
            kalman_params=kalman_params,
            joint_names=joint_names,
            keypoints=keypoints,
            edges=edges,
            defaults=runtime_defaults,
        )
        runtime_cfg["render_conf_min"] = float(
            np.clip(float(runtime_cfg["render_conf_min"]) + render_conf_min_offset, 0.0, 1.0)
        )
        runtime_cfg["update_conf_min"] = float(
            np.clip(float(runtime_cfg["update_conf_min"]) + update_conf_min_offset, 0.0, 1.0)
        )
        runtime_cfg["gate_threshold"] = float(max(1e-6, float(runtime_cfg["gate_threshold"]) + gate_threshold_offset))
        runtime_cfg["max_pos_std"] = float(max(1e-6, float(runtime_cfg["max_pos_std"]) + max_pos_std_offset))
        runtime_cfg["draw_conf_thresh"] = float(
            np.clip(float(runtime_cfg["draw_conf_thresh"]) + draw_conf_thresh_offset, 0.0, 1.0)
        )
        for joint_cfg in per_joint_cfg:
            if "render_conf_min" in joint_cfg:
                joint_cfg["render_conf_min"] = float(
                    np.clip(float(joint_cfg["render_conf_min"]) + render_conf_min_offset, 0.0, 1.0)
                )
            if "update_conf_min" in joint_cfg:
                joint_cfg["update_conf_min"] = float(
                    np.clip(float(joint_cfg["update_conf_min"]) + update_conf_min_offset, 0.0, 1.0)
                )
            if "gate_threshold" in joint_cfg:
                joint_cfg["gate_threshold"] = float(max(1e-6, float(joint_cfg["gate_threshold"]) + gate_threshold_offset))
            if "max_pos_std" in joint_cfg:
                joint_cfg["max_pos_std"] = float(max(1e-6, float(joint_cfg["max_pos_std"]) + max_pos_std_offset))

        tracker = SkeletonKalmanTracker(
            n_joints=keypoints.shape[1],
            edges=edges,
            process_var=float(runtime_cfg["process_var"]),
            obs_var=float(runtime_cfg["obs_var"]),
            gate_threshold=float(runtime_cfg["gate_threshold"]),
            velocity_damping=float(runtime_cfg["velocity_damping"]),
            max_missed_updates=int(runtime_cfg["max_missed_updates"]),
            reinit_confidence=float(runtime_cfg["reinit_confidence"]),
            bone_length_tol=float(runtime_cfg["bone_length_tol"]),
            bone_adjust_alpha=float(runtime_cfg["bone_adjust_alpha"]),
            bone_confidence_anchor=float(runtime_cfg["bone_confidence_anchor"]),
            update_conf_min=float(runtime_cfg["update_conf_min"]),
            render_conf_min=float(runtime_cfg["render_conf_min"]),
            max_extrapolation_frames=int(runtime_cfg["max_extrapolation_frames"]),
            max_pos_std=float(runtime_cfg["max_pos_std"]),
            per_joint_params=per_joint_cfg,
        )
        smoothed, visible_mask, effective_conf = tracker.smooth(keypoints)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_dir = Path(destfolder).resolve() if destfolder else video_path.parent.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        shuffle_tag = f"_shuffle{shuffle}" if shuffle is not None else ""
        out_path = out_dir / f"{video_path.stem}{shuffle_tag}_kalman.mp4"

        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 30.0,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise IOError(f"Could not create output video: {out_path}")

        frame_idx = 0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_video_frames <= 0:
            total_video_frames = None
        total_pose_frames = smoothed.shape[0]
        with tqdm(
            total=total_video_frames,
            desc=f"Kalman render: {video_path.name}",
            unit="frame",
            leave=True,
        ) as pbar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx < total_pose_frames:
                    frame = _draw_pose(
                        frame,
                        smoothed[frame_idx],
                        edges,
                        visible_mask[frame_idx],
                        effective_conf[frame_idx],
                        conf_thresh=float(runtime_cfg["draw_conf_thresh"]),
                    )
                writer.write(frame)
                frame_idx += 1
                pbar.update(1)

        writer.release()
        cap.release()
        output_videos.append(str(out_path))

    return output_videos
