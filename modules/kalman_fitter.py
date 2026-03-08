from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import deeplabcut
import numpy as np
import pandas as pd
import cv2

from modules.kalman import (
    _build_skeleton_edges,
    _estimate_scale_from_keypoints,
    _find_collected_h5,
    _fit_dispersion_threshold,
    _fit_joint_params_from_stats,
    _frame_number_from_index,
    _index_to_stem_image_key,
    _load_keypoints_from_df,
    _run_model_predictions_on_images,
    _subsample_indices,
)

try:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
except Exception:  # pragma: no cover - optional dependency
    ExtraTreesClassifier = None
    GradientBoostingClassifier = None
    HistGradientBoostingClassifier = None
    KNeighborsClassifier = None
    LinearSVC = None
    LogisticRegression = None
    RandomForestClassifier = None

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _collect_fit_data(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
) -> dict[str, Any]:
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
    cache_dir.mkdir(parents=True, exist_ok=True)

    sampled_records: list[dict[str, Any]] = []
    all_image_paths: list[Path] = []
    videos_used: list[str] = []
    video_scales: list[float] = []

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
        kept_rows: list[int] = []
        for ridx, idx_item in enumerate(sampled_index):
            img_name = Path(str(idx_item[-1] if isinstance(idx_item, tuple) else idx_item)).name
            img_path = (label_dir / img_name).resolve()
            if not img_path.exists():
                continue
            image_paths.append(img_path)
            image_keys.append(f"{stem}/{img_path.name}".lower())
            kept_rows.append(ridx)
        if not image_paths:
            continue

        keep = np.asarray(kept_rows, dtype=np.int64)
        frame_numbers = frame_numbers[keep]
        keypoints = keypoints[keep]

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
                "frame_numbers": frame_numbers,
                "gt_xy": keypoints[:, :, :2],
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
    pred_lookup = {_index_to_stem_image_key(idx): pred_keypoints[i] for i, idx in enumerate(pred_df.index)}

    speed_by_joint: dict[str, list[float]] = defaultdict(list)
    labels_by_joint: dict[str, list[int]] = defaultdict(list)
    likelihood_by_joint: dict[str, list[float]] = defaultdict(list)
    temporal_prev_by_joint: dict[str, list[float]] = defaultdict(list)
    temporal_next_by_joint: dict[str, list[float]] = defaultdict(list)
    frame_mean_by_joint: dict[str, list[float]] = defaultdict(list)
    rel_conf_by_joint: dict[str, list[float]] = defaultdict(list)
    motion_by_joint: dict[str, list[float]] = defaultdict(list)
    bone_med_by_joint: dict[str, list[float]] = defaultdict(list)
    bone_max_by_joint: dict[str, list[float]] = defaultdict(list)
    neighbor_conf_by_joint: dict[str, list[float]] = defaultdict(list)
    missing_by_joint: dict[str, int] = defaultdict(int)
    total_by_joint: dict[str, int] = defaultdict(int)

    for rec in sampled_records:
        joint_names = rec["joint_names"]
        frame_numbers = np.asarray(rec["frame_numbers"], dtype=np.int64)
        gt_xy = rec["gt_xy"]
        image_keys = rec["image_keys"]
        video_scale = float(rec["scale"])
        n_frames = len(image_keys)
        n_joints = len(joint_names)

        pred_for_video = np.full((n_frames, n_joints, 3), np.nan, dtype=np.float64)
        valid_rows = np.zeros(n_frames, dtype=bool)
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

        conf_mat = _clip01(pred_for_video[:, :, 2])
        xy_mat = pred_for_video[:, :, :2]
        with np.errstate(invalid="ignore"):
            frame_mean_conf = np.nanmean(conf_mat, axis=1)

        prev_conf = np.full_like(conf_mat, np.nan)
        next_conf = np.full_like(conf_mat, np.nan)
        prev_conf[1:] = conf_mat[:-1]
        next_conf[:-1] = conf_mat[1:]

        dt_prev = np.full(n_frames, np.nan, dtype=np.float64)
        dt_next = np.full(n_frames, np.nan, dtype=np.float64)
        dt_prev[1:] = np.diff(frame_numbers.astype(np.float64))
        dt_next[:-1] = np.diff(frame_numbers.astype(np.float64))
        delta_xy = np.linalg.norm(xy_mat[1:] - xy_mat[:-1], axis=2)
        speed_prev = np.full((n_frames, n_joints), np.nan, dtype=np.float64)
        speed_next = np.full((n_frames, n_joints), np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            speed_prev[1:] = (delta_xy / np.maximum(dt_prev[1:, None], 1.0)) / max(video_scale, 1e-6)
            speed_next[:-1] = (delta_xy / np.maximum(dt_next[:-1, None], 1.0)) / max(video_scale, 1e-6)
            motion_norm = np.nanmean(np.stack([speed_prev, speed_next], axis=2), axis=2)

        adjacency: list[list[int]] = [[] for _ in range(n_joints)]
        for a, b in rec["edges"]:
            if 0 <= a < n_joints and 0 <= b < n_joints:
                adjacency[a].append(b)
                adjacency[b].append(a)
        ref_lengths: dict[tuple[int, int], float] = {}
        for a, b in rec["edges"]:
            if not (0 <= a < n_joints and 0 <= b < n_joints):
                continue
            pa = gt_xy[:, a]
            pb = gt_xy[:, b]
            vis = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
            if not np.any(vis):
                continue
            length = float(np.median(np.linalg.norm(pa[vis] - pb[vis], axis=1)))
            if np.isfinite(length) and length > 1e-6:
                ref_lengths[(a, b)] = length
                ref_lengths[(b, a)] = length

        with np.errstate(invalid="ignore"):
            neighbor_conf = np.full((n_frames, n_joints), np.nan, dtype=np.float64)
            for j in range(n_joints):
                nbr = adjacency[j]
                if nbr:
                    neighbor_conf[:, j] = np.nanmean(conf_mat[:, nbr], axis=1)
                else:
                    neighbor_conf[:, j] = frame_mean_conf

        bone_med = np.full((n_frames, n_joints), np.nan, dtype=np.float64)
        bone_max = np.full((n_frames, n_joints), np.nan, dtype=np.float64)
        for j in range(n_joints):
            errs: list[np.ndarray] = []
            for nbr in adjacency[j]:
                ref = ref_lengths.get((j, nbr))
                if ref is None or ref <= 1e-6:
                    continue
                dist = np.linalg.norm(xy_mat[:, j] - xy_mat[:, nbr], axis=1)
                err = np.abs(dist - ref) / ref
                errs.append(err)
            if not errs:
                continue
            stack = np.vstack(errs)
            with np.errstate(invalid="ignore"):
                bone_med[:, j] = np.nanmedian(stack, axis=0)
                bone_max[:, j] = np.nanmax(stack, axis=0)

        for j, joint_name in enumerate(joint_names):
            joint_key = joint_name.split("|")[-1]
            gt_joint = gt_xy[:, j]
            gt_visible = np.isfinite(gt_joint).all(axis=1)
            pred_joint = pred_for_video[:, j]
            pred_visible = np.isfinite(pred_joint[:, :2]).all(axis=1)
            pred_conf = conf_mat[:, j]

            total_by_joint[joint_key] += int(gt_visible.size)
            missing_by_joint[joint_key] += int(gt_visible.size - np.count_nonzero(gt_visible))

            cls_mask = valid_rows & np.isfinite(pred_conf)
            if np.any(cls_mask):
                rel_conf = _clip01(np.clip(pred_conf / np.maximum(frame_mean_conf, 1e-6), 0.0, 3.0) / 3.0)
                labels_by_joint[joint_key].extend(gt_visible[cls_mask].astype(np.int32).tolist())
                likelihood_by_joint[joint_key].extend(pred_conf[cls_mask].astype(np.float64).tolist())
                temporal_prev_by_joint[joint_key].extend(prev_conf[:, j][cls_mask].astype(np.float64).tolist())
                temporal_next_by_joint[joint_key].extend(next_conf[:, j][cls_mask].astype(np.float64).tolist())
                frame_mean_by_joint[joint_key].extend(frame_mean_conf[cls_mask].astype(np.float64).tolist())
                rel_conf_by_joint[joint_key].extend(rel_conf[cls_mask].astype(np.float64).tolist())
                motion_by_joint[joint_key].extend(motion_norm[:, j][cls_mask].astype(np.float64).tolist())
                bone_med_by_joint[joint_key].extend(bone_med[:, j][cls_mask].astype(np.float64).tolist())
                bone_max_by_joint[joint_key].extend(bone_max[:, j][cls_mask].astype(np.float64).tolist())
                neighbor_conf_by_joint[joint_key].extend(neighbor_conf[:, j][cls_mask].astype(np.float64).tolist())

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
            take = speed_norm
            if take.size > max_speed_samples_per_joint:
                idx = _subsample_indices(take.size, max_speed_samples_per_joint)
                take = take[idx]
            speed_by_joint[joint_key].extend(take.astype(np.float64).tolist())

    if not total_by_joint:
        raise ValueError("No matched prediction/label pairs were found for fitting.")

    return {
        "cfg": cfg,
        "videos_used": sorted(set(videos_used)),
        "video_scales": video_scales,
        "speed_by_joint": speed_by_joint,
        "labels_by_joint": labels_by_joint,
        "likelihood_by_joint": likelihood_by_joint,
        "prev_by_joint": temporal_prev_by_joint,
        "next_by_joint": temporal_next_by_joint,
        "frame_mean_by_joint": frame_mean_by_joint,
        "rel_conf_by_joint": rel_conf_by_joint,
        "motion_by_joint": motion_by_joint,
        "bone_med_by_joint": bone_med_by_joint,
        "bone_max_by_joint": bone_max_by_joint,
        "neighbor_conf_by_joint": neighbor_conf_by_joint,
        "missing_by_joint": missing_by_joint,
        "total_by_joint": total_by_joint,
    }

def _joint_raw_arrays(data: dict[str, Any], joint_key: str) -> dict[str, np.ndarray]:
    return {
        "likelihood": np.asarray(data["likelihood_by_joint"].get(joint_key, []), dtype=np.float64),
        "prev": np.asarray(data["prev_by_joint"].get(joint_key, []), dtype=np.float64),
        "next": np.asarray(data["next_by_joint"].get(joint_key, []), dtype=np.float64),
        "frame_mean": np.asarray(data["frame_mean_by_joint"].get(joint_key, []), dtype=np.float64),
        "rel_conf": np.asarray(data["rel_conf_by_joint"].get(joint_key, []), dtype=np.float64),
        "motion": np.asarray(data["motion_by_joint"].get(joint_key, []), dtype=np.float64),
        "bone_med": np.asarray(data["bone_med_by_joint"].get(joint_key, []), dtype=np.float64),
        "bone_max": np.asarray(data["bone_max_by_joint"].get(joint_key, []), dtype=np.float64),
        "neighbor_conf": np.asarray(data["neighbor_conf_by_joint"].get(joint_key, []), dtype=np.float64),
    }


def _build_joint_features(raw: dict[str, np.ndarray]) -> np.ndarray:
    curr = _clip01(raw["likelihood"])
    prev = _clip01(raw["prev"])
    nxt = _clip01(raw["next"])
    with np.errstate(invalid="ignore"):
        nei = np.nanmean(np.vstack([prev, nxt]), axis=0)
    nei = np.where(np.isfinite(nei), nei, curr)
    rel = _clip01(raw["rel_conf"])
    fm = _clip01(raw["frame_mean"])
    neigh = _clip01(raw["neighbor_conf"])
    motion = np.clip(np.nan_to_num(raw["motion"], nan=0.8, posinf=0.8, neginf=0.8), 0.0, 3.0)
    bone_med = np.clip(np.nan_to_num(raw["bone_med"], nan=0.9, posinf=0.9, neginf=0.9), 0.0, 3.0)
    bone_max = np.clip(np.nan_to_num(raw["bone_max"], nan=1.2, posinf=1.2, neginf=1.2), 0.0, 3.0)
    low = np.minimum(curr, nei)
    harm = (2.0 * curr * nei) / np.maximum(curr + nei, 1e-6)
    exp_motion = np.exp(-1.7 * motion)
    exp_bone = np.exp(-1.9 * bone_med)
    return np.column_stack(
        [curr, nei, rel, fm, neigh, motion, bone_med, bone_max, low, harm, exp_motion, exp_bone]
    )


def _predict_visible_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x)[:, 1]
    else:
        raw_score = model.decision_function(x)
        score = 1.0 / (1.0 + np.exp(-raw_score))
    return _clip01(np.asarray(score, dtype=np.float64))


def _resolve_torch_device(torch_device: str | None = None) -> Any:
    if torch is None:
        return None
    if torch_device:
        dev = torch.device(torch_device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return dev
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _crop_patch_gray(image_gray: np.ndarray, x: float, y: float, patch_size: int) -> np.ndarray:
    half = int(patch_size // 2)
    h, w = image_gray.shape[:2]
    cx = int(round(float(x)))
    cy = int(round(float(y)))
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + patch_size
    y1 = y0 + patch_size

    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        image_gray = cv2.copyMakeBorder(
            image_gray,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        x0 += pad_l
        y0 += pad_t
    patch = image_gray[y0 : y0 + patch_size, x0 : x0 + patch_size]
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    return patch.astype(np.float32) / 255.0


def _collect_patch_samples(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    use_cached_predictions: bool = True,
    patch_size: int = 48,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    project_root = Path(project_path).resolve()
    config_path = project_root / "config.yaml"
    cfg = deeplabcut.auxiliaryfunctions.read_config(str(config_path))
    video_sets = cfg.get("video_sets", {})
    scorer = str(cfg.get("scorer", "")).strip()
    labeled_root = project_root / "labeled-data"
    cache_dir = project_root / ".kalman_fit_cache" / f"shuffle{shuffle}"
    cache_dir.mkdir(parents=True, exist_ok=True)

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
        keypoints = keypoints[sample_idx]
        sampled_index = np.array(df.index, dtype=object)[valid_frame_idx][order][sample_idx]

        image_paths: list[Path] = []
        image_keys: list[str] = []
        kept_rows: list[int] = []
        for ridx, idx_item in enumerate(sampled_index):
            img_name = Path(str(idx_item[-1] if isinstance(idx_item, tuple) else idx_item)).name
            img_path = (label_dir / img_name).resolve()
            if not img_path.exists():
                continue
            image_paths.append(img_path)
            image_keys.append(f"{stem}/{img_path.name}".lower())
            kept_rows.append(ridx)
        if not image_paths:
            continue

        keep = np.asarray(kept_rows, dtype=np.int64)
        keypoints = keypoints[keep]
        sampled_records.append(
            {
                "joint_names": joint_names,
                "gt_xy": keypoints[:, :, :2],
                "image_keys": image_keys,
                "image_paths": image_paths,
            }
        )
        all_image_paths.extend(image_paths)

    if not sampled_records:
        raise ValueError("No sampled labeled images for patch classifier.")

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
    pred_lookup = {_index_to_stem_image_key(idx): pred_keypoints[i] for i, idx in enumerate(pred_df.index)}

    bodyparts = [str(bp) for bp in cfg.get("bodyparts", [])]
    fitted_joint_names = bodyparts if bodyparts else sorted({jn.split("|")[-1] for rec in sampled_records for jn in rec["joint_names"]})
    patches_by_joint: dict[str, list[np.ndarray]] = defaultdict(list)
    labels_by_joint: dict[str, list[int]] = defaultdict(list)
    conf_by_joint: dict[str, list[float]] = defaultdict(list)

    for rec in sampled_records:
        joint_names = rec["joint_names"]
        gt_xy = rec["gt_xy"]
        image_keys = rec["image_keys"]
        image_paths = rec["image_paths"]
        pred_for_video = np.full((len(image_keys), len(joint_names), 3), np.nan, dtype=np.float64)
        valid_rows = np.zeros(len(image_keys), dtype=bool)
        for ridx, key in enumerate(image_keys):
            row = pred_lookup.get(key)
            if row is None:
                continue
            valid_rows[ridx] = True
            for j, jn in enumerate(joint_names):
                pidx = pred_joint_idx.get(jn.split("|")[-1])
                if pidx is not None:
                    pred_for_video[ridx, j] = row[pidx]

        for t, img_path in enumerate(image_paths):
            if not valid_rows[t]:
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            for j, jn in enumerate(joint_names):
                key = jn.split("|")[-1]
                if key not in fitted_joint_names:
                    continue
                x, y, c = pred_for_video[t, j]
                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)):
                    continue
                label = int(np.isfinite(gt_xy[t, j]).all())
                patch = _crop_patch_gray(img, x, y, patch_size=patch_size)
                patches_by_joint[key].append(patch)
                labels_by_joint[key].append(label)
                conf_by_joint[key].append(float(np.clip(c, 0.0, 1.0)))

    patch_arr = {k: np.stack(v, axis=0).astype(np.float32) if v else np.empty((0, patch_size, patch_size), dtype=np.float32) for k, v in patches_by_joint.items()}
    label_arr = {k: np.asarray(v, dtype=np.int32) for k, v in labels_by_joint.items()}
    conf_arr = {k: np.asarray(v, dtype=np.float32) for k, v in conf_by_joint.items()}
    return fitted_joint_names, patch_arr, label_arr, conf_arr


class _TinyVisibilityCNN(nn.Module):
    def __init__(self, patch_size: int, n_joints: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        feat_hw = max(1, patch_size // 8)
        self.fc_img = nn.Linear(64 * feat_hw * feat_hw, 64)
        self.joint_emb = nn.Embedding(n_joints, 8)
        self.head = nn.Sequential(
            nn.Linear(64 + 8 + 1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, patch: torch.Tensor, joint_id: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        x = self.conv(patch)
        x = x.flatten(1)
        x = torch.relu(self.fc_img(x))
        j = self.joint_emb(joint_id)
        c = conf.unsqueeze(1)
        z = torch.cat([x, j, c], dim=1)
        return self.head(z).squeeze(1)


def _build_joint_split(
    labels_by_joint: dict[str, list[int]] | dict[str, np.ndarray],
    joint_keys: list[str],
    test_fraction: float,
    test_size: int | None,
    test_seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(int(test_seed))
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    frac = float(np.clip(test_fraction, 0.01, 0.8))
    for key in joint_keys:
        y = np.asarray(labels_by_joint.get(key, []), dtype=np.int32)
        n = y.size
        if n <= 1:
            splits[key] = (np.arange(n, dtype=np.int64), np.array([], dtype=np.int64))
            continue

        if test_size is None:
            n_test = int(round(n * frac))
        else:
            n_test = int(test_size)
        n_test = int(np.clip(n_test, 1, max(1, n - 1)))

        cls0 = np.where(y == 0)[0]
        cls1 = np.where(y == 1)[0]
        if cls0.size > 0 and cls1.size > 0:
            # Stratified split by visibility label.
            n0 = int(round(n_test * (cls0.size / n)))
            n0 = int(np.clip(n0, 1, max(1, cls0.size - 1)))
            n1 = n_test - n0
            n1 = int(np.clip(n1, 1, max(1, cls1.size - 1)))
            while n0 + n1 > n_test:
                if n0 > n1 and n0 > 1:
                    n0 -= 1
                elif n1 > 1:
                    n1 -= 1
                else:
                    break
            while n0 + n1 < n_test:
                if n0 < cls0.size - 1:
                    n0 += 1
                elif n1 < cls1.size - 1:
                    n1 += 1
                else:
                    break
            test_idx = np.concatenate([rng.choice(cls0, size=n0, replace=False), rng.choice(cls1, size=n1, replace=False)])
        else:
            test_idx = rng.choice(np.arange(n), size=n_test, replace=False)
        test_idx = np.unique(test_idx.astype(np.int64))
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0].astype(np.int64)
        if train_idx.size == 0:
            train_idx = np.array([int(test_idx[0])], dtype=np.int64)
            test_idx = test_idx[1:]
        splits[key] = (train_idx, test_idx)
    return splits


def _build_frame_split(
    n_samples: int,
    test_fraction: float,
    test_size: int | None,
    test_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples <= 1:
        return np.arange(n_samples, dtype=np.int64), np.array([], dtype=np.int64)
    if test_size is None:
        n_test = int(round(n_samples * float(np.clip(test_fraction, 0.01, 0.8))))
    else:
        n_test = int(test_size)
    n_test = int(np.clip(n_test, 1, max(1, n_samples - 1)))
    rng = np.random.default_rng(int(test_seed))
    test_idx = np.sort(rng.choice(np.arange(n_samples), size=n_test, replace=False).astype(np.int64))
    train_mask = np.ones(n_samples, dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.where(train_mask)[0].astype(np.int64)
    if train_idx.size == 0:
        train_idx = np.array([int(test_idx[0])], dtype=np.int64)
        test_idx = test_idx[1:]
    return train_idx, test_idx


def _evaluate_on_split(
    fitted_joint_names: list[str],
    per_joint_thresholds: dict[str, float],
    global_threshold: float,
    labels_eval_by_joint: dict[str, np.ndarray],
    scores_eval_by_joint: dict[str, np.ndarray],
) -> dict[str, Any]:
    per_joint_vis: dict[str, float] = {}
    per_joint_invis: dict[str, float] = {}
    all_labels: list[np.ndarray] = []
    all_pred_vis: list[np.ndarray] = []
    for key in fitted_joint_names:
        y = np.asarray(labels_eval_by_joint.get(key, []), dtype=np.int32)
        s = np.asarray(scores_eval_by_joint.get(key, []), dtype=np.float64)
        n = min(y.size, s.size)
        if n == 0:
            continue
        y = y[:n]
        s = s[:n]
        t = float(per_joint_thresholds.get(key, global_threshold))
        pred_vis = s >= t
        tp = float(np.sum(pred_vis & (y == 1)))
        fn = float(np.sum((~pred_vis) & (y == 1)))
        tn = float(np.sum((~pred_vis) & (y == 0)))
        fp = float(np.sum(pred_vis & (y == 0)))
        per_joint_vis[key] = tp / max(1.0, tp + fn)
        per_joint_invis[key] = tn / max(1.0, tn + fp)
        all_labels.append(y)
        all_pred_vis.append(pred_vis.astype(bool))

    if all_labels:
        y_all = np.concatenate(all_labels, axis=0)
        p_all = np.concatenate(all_pred_vis, axis=0)
        tp = float(np.sum(p_all & (y_all == 1)))
        fn = float(np.sum((~p_all) & (y_all == 1)))
        tn = float(np.sum((~p_all) & (y_all == 0)))
        fp = float(np.sum(p_all & (y_all == 0)))
        global_vis = tp / max(1.0, tp + fn)
        global_invis = tn / max(1.0, tn + fp)
    else:
        global_vis = 0.0
        global_invis = 0.0

    return {
        "split": "test",
        "global_visible_recall": float(np.clip(global_vis, 0.0, 1.0)),
        "global_invisible_recall": float(np.clip(global_invis, 0.0, 1.0)),
        "per_joint_visible_recall": {k: float(np.clip(v, 0.0, 1.0)) for k, v in per_joint_vis.items()},
        "per_joint_invisible_recall": {k: float(np.clip(v, 0.0, 1.0)) for k, v in per_joint_invis.items()},
    }


def _finalize_fit(
    data: dict[str, Any],
    per_joint_scores: dict[str, np.ndarray],
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int,
    dispersion_visible_recall_target: float,
    score_model_name: str,
    labels_fit_by_joint: dict[str, np.ndarray] | None = None,
    scores_eval_by_joint: dict[str, np.ndarray] | None = None,
    labels_eval_by_joint: dict[str, np.ndarray] | None = None,
    cutoff_split: str = "test",
) -> tuple[dict[str, Any], dict[str, float]]:
    cfg = data["cfg"]
    speed_by_joint = data["speed_by_joint"]
    labels_by_joint_raw = data["labels_by_joint"]
    missing_by_joint = data["missing_by_joint"]
    total_by_joint = data["total_by_joint"]

    bodyparts = [str(bp) for bp in cfg.get("bodyparts", [])]
    fitted_joint_names = bodyparts if bodyparts else sorted(total_by_joint.keys())

    all_speed = (
        np.concatenate([np.asarray(v, dtype=np.float64) for v in speed_by_joint.values() if len(v)], axis=0)
        if any(len(v) for v in speed_by_joint.values())
        else np.array([], dtype=np.float64)
    )
    if labels_fit_by_joint is None:
        labels_fit = {k: np.asarray(v, dtype=np.int32) for k, v in labels_by_joint_raw.items()}
    else:
        labels_fit = {k: np.asarray(v, dtype=np.int32) for k, v in labels_fit_by_joint.items()}

    use_eval_cutoff = (
        str(cutoff_split).lower() == "test"
        and labels_eval_by_joint is not None
        and scores_eval_by_joint is not None
    )
    labels_for_cutoff = labels_eval_by_joint if use_eval_cutoff else labels_fit
    scores_for_cutoff = scores_eval_by_joint if use_eval_cutoff else per_joint_scores

    all_labels = (
        np.concatenate([np.asarray(v, dtype=np.int32) for v in labels_for_cutoff.values() if len(v)], axis=0)
        if any(len(v) for v in labels_for_cutoff.values())
        else np.array([], dtype=np.int32)
    )
    all_scores = (
        np.concatenate([np.asarray(v, dtype=np.float64) for v in scores_for_cutoff.values() if len(v)], axis=0)
        if any(len(v) for v in scores_for_cutoff.values())
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
    )
    global_disp_threshold, global_vis_recall, global_invis_recall = _fit_dispersion_threshold(
        visibility_labels=all_labels,
        dispersion_scores=all_scores,
        visible_recall_target=dispersion_visible_recall_target,
    )

    per_joint: dict[str, dict[str, Any]] = {}
    per_joint_invisible_recall: dict[str, float] = {}
    for joint_key in fitted_joint_names:
        speeds = np.asarray(speed_by_joint.get(joint_key, []), dtype=np.float64)
        if speeds.size == 0 and all_speed.size:
            speeds = all_speed
        miss_ratio = float(missing_by_joint.get(joint_key, 0) / max(1, total_by_joint.get(joint_key, 0)))
        if total_by_joint.get(joint_key, 0) == 0:
            miss_ratio = global_missing_ratio

        labels = np.asarray(labels_for_cutoff.get(joint_key, []), dtype=np.int32)
        scores = np.asarray(scores_for_cutoff.get(joint_key, []), dtype=np.float64)
        if labels.size == 0 and all_labels.size:
            labels = all_labels
            scores = all_scores
        else:
            n = min(labels.size, scores.size)
            labels = labels[:n]
            scores = scores[:n]

        base = _fit_joint_params_from_stats(speed_samples=speeds, missing_ratio=miss_ratio)
        if labels.size and scores.size:
            t, _, invis = _fit_dispersion_threshold(
                visibility_labels=labels,
                dispersion_scores=scores,
                visible_recall_target=dispersion_visible_recall_target,
            )
        else:
            t, invis = global_disp_threshold, global_invis_recall
        base["dispersion_threshold"] = float(np.clip(t, 0.0, 1.0))
        per_joint[joint_key] = base
        per_joint_invisible_recall[joint_key] = float(np.clip(invis, 0.0, 1.0))

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
        "dispersion_visible_recall_target": float(np.clip(dispersion_visible_recall_target, 0.5, 0.999)),
        "dispersion_threshold": float(np.clip(global_disp_threshold, 0.0, 1.0)),
        "dispersion_visible_recall": float(np.clip(global_vis_recall, 0.0, 1.0)),
        "dispersion_invisible_recall": float(np.clip(global_invis_recall, 0.0, 1.0)),
        "dispersion_score_model": score_model_name,
        "dispersion_cutoff_split": "test" if use_eval_cutoff else "train",
    }

    kalman_params = {
        "version": 4,
        "project_path": str(Path(project_path).resolve()),
        "shuffle": int(shuffle),
        "trainingsetindex": int(trainingsetindex),
        "videos_used": data["videos_used"],
        "scale_reference": float(np.median(data["video_scales"])) if data["video_scales"] else 200.0,
        "global": global_defaults,
        "per_joint": per_joint,
    }
    per_joint_out = per_joint_invisible_recall
    if labels_eval_by_joint is not None and scores_eval_by_joint is not None:
        per_joint_thresholds = {
            k: float(v.get("dispersion_threshold", global_disp_threshold)) for k, v in per_joint.items()
        }
        eval_metrics = _evaluate_on_split(
            fitted_joint_names=fitted_joint_names,
            per_joint_thresholds=per_joint_thresholds,
            global_threshold=float(global_disp_threshold),
            labels_eval_by_joint=labels_eval_by_joint,
            scores_eval_by_joint=scores_eval_by_joint,
        )
        eval_metrics["target_visible_recall"] = float(np.clip(dispersion_visible_recall_target, 0.5, 0.999))
        eval_metrics["cutoff_split"] = "test" if use_eval_cutoff else "train"
        kalman_params["evaluation"] = eval_metrics
        per_joint_eval_invis = eval_metrics.get("per_joint_invisible_recall", {})
        if per_joint_eval_invis:
            per_joint_out = {k: float(v) for k, v in per_joint_eval_invis.items()}

    return kalman_params, per_joint_out

def _fit_from_score_builder(
    project_path: str | Path,
    shuffle: int,
    score_builder: Callable[[dict[str, np.ndarray]], np.ndarray],
    score_model_name: str,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    data = _collect_fit_data(
        project_path=project_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
    )

    bodyparts = [str(bp) for bp in data["cfg"].get("bodyparts", [])]
    fitted_joint_names = bodyparts if bodyparts else sorted(data["total_by_joint"].keys())
    split = _build_joint_split(
        labels_by_joint=data["labels_by_joint"],
        joint_keys=fitted_joint_names,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )
    per_joint_scores_fit: dict[str, np.ndarray] = {}
    labels_fit_by_joint: dict[str, np.ndarray] = {}
    per_joint_scores_eval: dict[str, np.ndarray] = {}
    labels_eval_by_joint: dict[str, np.ndarray] = {}
    for key in fitted_joint_names:
        raw = _joint_raw_arrays(data, key)
        labels = np.asarray(data["labels_by_joint"].get(key, []), dtype=np.int32)
        n = min(labels.size, raw["likelihood"].size)
        labels = labels[:n]
        for rk, rv in raw.items():
            raw[rk] = rv[:n]
        scores = _clip01(np.asarray(score_builder(raw), dtype=np.float64))

        train_idx, test_idx = split.get(key, (np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)))
        train_idx = train_idx[train_idx < n]
        test_idx = test_idx[test_idx < n]
        per_joint_scores_fit[key] = scores[train_idx]
        labels_fit_by_joint[key] = labels[train_idx]
        per_joint_scores_eval[key] = scores[test_idx]
        labels_eval_by_joint[key] = labels[test_idx]

    return _finalize_fit(
        data=data,
        per_joint_scores=per_joint_scores_fit,
        project_path=project_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        score_model_name=score_model_name,
        labels_fit_by_joint=labels_fit_by_joint,
        scores_eval_by_joint=per_joint_scores_eval,
        labels_eval_by_joint=labels_eval_by_joint,
    )


def _fit_from_ml_model(
    project_path: str | Path,
    shuffle: int,
    score_model_name: str,
    model_factory: Callable[[], Any],
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    data = _collect_fit_data(
        project_path=project_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
    )
    labels_by_joint = data["labels_by_joint"]

    bodyparts = [str(bp) for bp in data["cfg"].get("bodyparts", [])]
    fitted_joint_names = bodyparts if bodyparts else sorted(data["total_by_joint"].keys())
    split = _build_joint_split(
        labels_by_joint=labels_by_joint,
        joint_keys=fitted_joint_names,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )
    per_joint_scores_fit: dict[str, np.ndarray] = {}
    labels_fit_by_joint: dict[str, np.ndarray] = {}
    per_joint_scores_eval: dict[str, np.ndarray] = {}
    labels_eval_by_joint: dict[str, np.ndarray] = {}

    pooled_x: list[np.ndarray] = []
    pooled_y: list[np.ndarray] = []
    joint_payload: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for joint_key in fitted_joint_names:
        raw = _joint_raw_arrays(data, joint_key)
        y = np.asarray(labels_by_joint.get(joint_key, []), dtype=np.int32)
        n = min(y.size, raw["likelihood"].size)
        y = y[:n]
        for rk, rv in raw.items():
            raw[rk] = rv[:n]
        x = _build_joint_features(raw) if n > 0 else np.empty((0, 12), dtype=np.float64)
        base_score = _clip01(raw["likelihood"])
        train_idx, test_idx = split.get(joint_key, (np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)))
        train_idx = train_idx[train_idx < n]
        test_idx = test_idx[test_idx < n]
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]
        joint_payload[joint_key] = (x_train, y_train, x_test, y_test, base_score)
        if y_train.size > 0 and np.unique(y_train).size >= 2:
            pooled_x.append(x_train)
            pooled_y.append(y_train)

    global_model = None
    if pooled_x:
        global_x = np.concatenate(pooled_x, axis=0)
        global_y = np.concatenate(pooled_y, axis=0)
        try:
            global_model = model_factory()
            global_model.fit(global_x, global_y)
        except Exception:
            global_model = None

    for joint_key in fitted_joint_names:
        x_train, y_train, x_test, y_test, base_score = joint_payload[joint_key]
        if y_train.size == 0:
            per_joint_scores_fit[joint_key] = np.array([], dtype=np.float64)
            labels_fit_by_joint[joint_key] = np.array([], dtype=np.int32)
            per_joint_scores_eval[joint_key] = np.array([], dtype=np.float64)
            labels_eval_by_joint[joint_key] = y_test
            continue

        model = None
        if np.unique(y_train).size >= 2 and y_train.size >= 80:
            try:
                model = model_factory()
                model.fit(x_train, y_train)
            except Exception:
                model = None
        if model is None:
            model = global_model

        if model is None:
            train_idx, test_idx = split.get(joint_key, (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
            per_joint_scores_fit[joint_key] = base_score[train_idx]
            labels_fit_by_joint[joint_key] = y_train
            per_joint_scores_eval[joint_key] = base_score[test_idx]
            labels_eval_by_joint[joint_key] = y_test
            continue

        per_joint_scores_fit[joint_key] = _predict_visible_scores(model, x_train)
        labels_fit_by_joint[joint_key] = y_train
        per_joint_scores_eval[joint_key] = _predict_visible_scores(model, x_test) if x_test.size else np.array([], dtype=np.float64)
        labels_eval_by_joint[joint_key] = y_test

    return _finalize_fit(
        data=data,
        per_joint_scores=per_joint_scores_fit,
        project_path=project_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        score_model_name=score_model_name,
        labels_fit_by_joint=labels_fit_by_joint,
        scores_eval_by_joint=per_joint_scores_eval,
        labels_eval_by_joint=labels_eval_by_joint,
    )


def _fit_from_ml_ensemble(
    project_path: str | Path,
    shuffle: int,
    score_model_name: str,
    model_factories: list[Callable[[], Any]],
    model_weights: list[float] | None = None,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    data = _collect_fit_data(
        project_path=project_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
    )
    labels_by_joint = data["labels_by_joint"]

    bodyparts = [str(bp) for bp in data["cfg"].get("bodyparts", [])]
    fitted_joint_names = bodyparts if bodyparts else sorted(data["total_by_joint"].keys())
    n_models = len(model_factories)
    if n_models == 0:
        raise ValueError("model_factories must contain at least one model.")
    if model_weights is None:
        weights = np.ones(n_models, dtype=np.float64) / n_models
    else:
        weights = np.asarray(model_weights, dtype=np.float64)
        if weights.size != n_models:
            raise ValueError("model_weights size must match model_factories size.")
        weights = weights / max(np.sum(weights), 1e-9)

    split = _build_joint_split(
        labels_by_joint=labels_by_joint,
        joint_keys=fitted_joint_names,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )

    pooled_x: list[np.ndarray] = []
    pooled_y: list[np.ndarray] = []
    joint_payload: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for joint_key in fitted_joint_names:
        raw = _joint_raw_arrays(data, joint_key)
        y = np.asarray(labels_by_joint.get(joint_key, []), dtype=np.int32)
        n = min(y.size, raw["likelihood"].size)
        y = y[:n]
        for rk, rv in raw.items():
            raw[rk] = rv[:n]
        x = _build_joint_features(raw) if n > 0 else np.empty((0, 12), dtype=np.float64)
        base_score = _clip01(raw["likelihood"])
        train_idx, test_idx = split.get(joint_key, (np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)))
        train_idx = train_idx[train_idx < n]
        test_idx = test_idx[test_idx < n]
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]
        joint_payload[joint_key] = (x_train, y_train, x_test, y_test, base_score)
        if y_train.size > 0 and np.unique(y_train).size >= 2:
            pooled_x.append(x_train)
            pooled_y.append(y_train)

    global_models: list[Any | None] = []
    if pooled_x:
        global_x = np.concatenate(pooled_x, axis=0)
        global_y = np.concatenate(pooled_y, axis=0)
        for factory in model_factories:
            try:
                model = factory()
                model.fit(global_x, global_y)
                global_models.append(model)
            except Exception:
                global_models.append(None)
    else:
        global_models = [None] * n_models

    per_joint_scores_fit: dict[str, np.ndarray] = {}
    labels_fit_by_joint: dict[str, np.ndarray] = {}
    per_joint_scores_eval: dict[str, np.ndarray] = {}
    labels_eval_by_joint: dict[str, np.ndarray] = {}
    for joint_key in fitted_joint_names:
        x_train, y_train, x_test, y_test, base_score = joint_payload[joint_key]
        if y_train.size == 0:
            per_joint_scores_fit[joint_key] = np.array([], dtype=np.float64)
            labels_fit_by_joint[joint_key] = np.array([], dtype=np.int32)
            per_joint_scores_eval[joint_key] = np.array([], dtype=np.float64)
            labels_eval_by_joint[joint_key] = y_test
            continue

        per_model_scores_train: list[np.ndarray] = []
        per_model_scores_test: list[np.ndarray] = []
        for midx, factory in enumerate(model_factories):
            model = None
            if np.unique(y_train).size >= 2 and y_train.size >= 80:
                try:
                    model = factory()
                    model.fit(x_train, y_train)
                except Exception:
                    model = None
            if model is None:
                model = global_models[midx]
            if model is None:
                train_idx, test_idx = split.get(joint_key, (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
                per_model_scores_train.append(base_score[train_idx])
                per_model_scores_test.append(base_score[test_idx])
                continue
            per_model_scores_train.append(_predict_visible_scores(model, x_train))
            per_model_scores_test.append(
                _predict_visible_scores(model, x_test) if x_test.size else np.array([], dtype=np.float64)
            )

        stacked_train = np.vstack(per_model_scores_train)
        score_train = np.average(stacked_train, axis=0, weights=weights)
        per_joint_scores_fit[joint_key] = _clip01(score_train)
        labels_fit_by_joint[joint_key] = y_train

        if per_model_scores_test and per_model_scores_test[0].size:
            stacked_test = np.vstack(per_model_scores_test)
            score_test = np.average(stacked_test, axis=0, weights=weights)
            per_joint_scores_eval[joint_key] = _clip01(score_test)
        else:
            per_joint_scores_eval[joint_key] = np.array([], dtype=np.float64)
        labels_eval_by_joint[joint_key] = y_test

    return _finalize_fit(
        data=data,
        per_joint_scores=per_joint_scores_fit,
        project_path=project_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        score_model_name=score_model_name,
        labels_fit_by_joint=labels_fit_by_joint,
        scores_eval_by_joint=per_joint_scores_eval,
        labels_eval_by_joint=labels_eval_by_joint,
    )


def _fit_from_patch_cnn(
    project_path: str | Path,
    shuffle: int,
    score_model_name: str,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
    patch_size: int = 48,
    epochs: int = 6,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    torch_device: str | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for patch CNN visibility classifier.")

    data = _collect_fit_data(
        project_path=project_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
    )
    fitted_joint_names, patches_by_joint, labels_by_joint, conf_by_joint = _collect_patch_samples(
        project_path=project_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        use_cached_predictions=use_cached_predictions,
        patch_size=patch_size,
    )

    split = _build_joint_split(
        labels_by_joint=labels_by_joint,
        joint_keys=fitted_joint_names,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )

    joint_to_idx = {k: i for i, k in enumerate(fitted_joint_names)}
    x_train_list: list[np.ndarray] = []
    y_train_list: list[np.ndarray] = []
    j_train_list: list[np.ndarray] = []
    c_train_list: list[np.ndarray] = []
    x_test_by_joint: dict[str, np.ndarray] = {}
    y_test_by_joint: dict[str, np.ndarray] = {}
    c_test_by_joint: dict[str, np.ndarray] = {}

    for key in fitted_joint_names:
        p = patches_by_joint.get(key, np.empty((0, patch_size, patch_size), dtype=np.float32))
        y = labels_by_joint.get(key, np.empty((0,), dtype=np.int32))
        c = conf_by_joint.get(key, np.empty((0,), dtype=np.float32))
        n = min(len(p), len(y), len(c))
        p = p[:n]
        y = y[:n]
        c = c[:n]
        train_idx, test_idx = split.get(key, (np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)))
        train_idx = train_idx[train_idx < n]
        test_idx = test_idx[test_idx < n]
        if train_idx.size:
            x_train_list.append(p[train_idx])
            y_train_list.append(y[train_idx])
            c_train_list.append(c[train_idx])
            j_train_list.append(np.full(train_idx.size, joint_to_idx[key], dtype=np.int64))
        x_test_by_joint[key] = p[test_idx]
        y_test_by_joint[key] = y[test_idx]
        c_test_by_joint[key] = c[test_idx]

    if not x_train_list:
        raise ValueError("No train samples for patch CNN.")

    torch.manual_seed(int(test_seed))
    np.random.seed(int(test_seed))
    x_train = np.concatenate(x_train_list, axis=0)[:, None, :, :]
    y_train = np.concatenate(y_train_list, axis=0).astype(np.float32)
    j_train = np.concatenate(j_train_list, axis=0).astype(np.int64)
    c_train = np.concatenate(c_train_list, axis=0).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(j_train),
        torch.from_numpy(c_train),
        torch.from_numpy(y_train),
    )
    device = _resolve_torch_device(torch_device)
    loader = DataLoader(
        dataset,
        batch_size=int(max(16, batch_size)),
        shuffle=True,
        num_workers=0,
        pin_memory=bool(device and device.type == "cuda"),
    )
    model = _TinyVisibilityCNN(patch_size=patch_size, n_joints=len(fitted_joint_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    pos_weight = torch.tensor([max(1e-6, neg / max(pos, 1e-6))], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for _ in range(int(max(1, epochs))):
        for xb, jb, cb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            jb = jb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, jb, cb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    per_joint_scores_fit: dict[str, np.ndarray] = {}
    labels_fit_by_joint: dict[str, np.ndarray] = {}
    per_joint_scores_eval: dict[str, np.ndarray] = {}
    labels_eval_by_joint: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for key in fitted_joint_names:
            p = patches_by_joint.get(key, np.empty((0, patch_size, patch_size), dtype=np.float32))
            y = labels_by_joint.get(key, np.empty((0,), dtype=np.int32))
            c = conf_by_joint.get(key, np.empty((0,), dtype=np.float32))
            n = min(len(p), len(y), len(c))
            p = p[:n]
            y = y[:n]
            c = c[:n]
            train_idx, test_idx = split.get(key, (np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)))
            train_idx = train_idx[train_idx < n]
            test_idx = test_idx[test_idx < n]

            jid = torch.full((n,), joint_to_idx[key], dtype=torch.long)
            if n > 0:
                logits = model(
                    torch.from_numpy(p[:, None, :, :]).to(device, non_blocking=True),
                    jid.to(device, non_blocking=True),
                    torch.from_numpy(c.astype(np.float32)).to(device, non_blocking=True),
                )
                probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
            else:
                probs = np.array([], dtype=np.float64)

            per_joint_scores_fit[key] = probs[train_idx]
            labels_fit_by_joint[key] = y[train_idx]
            per_joint_scores_eval[key] = probs[test_idx]
            labels_eval_by_joint[key] = y[test_idx]

    return _finalize_fit(
        data=data,
        per_joint_scores=per_joint_scores_fit,
        project_path=project_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        score_model_name=score_model_name,
        labels_fit_by_joint=labels_fit_by_joint,
        scores_eval_by_joint=per_joint_scores_eval,
        labels_eval_by_joint=labels_eval_by_joint,
        cutoff_split="test",
    )


def _collect_roi_samples(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    use_cached_predictions: bool = True,
    roi_size: int = 96,
) -> tuple[
    list[str],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    project_root = Path(project_path).resolve()
    config_path = project_root / "config.yaml"
    cfg = deeplabcut.auxiliaryfunctions.read_config(str(config_path))
    video_sets = cfg.get("video_sets", {})
    scorer = str(cfg.get("scorer", "")).strip()
    labeled_root = project_root / "labeled-data"
    cache_dir = project_root / ".kalman_fit_cache" / f"shuffle{shuffle}"
    cache_dir.mkdir(parents=True, exist_ok=True)

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
        keypoints = keypoints[sample_idx]
        sampled_index = np.array(df.index, dtype=object)[valid_frame_idx][order][sample_idx]

        image_paths: list[Path] = []
        image_keys: list[str] = []
        kept_rows: list[int] = []
        for ridx, idx_item in enumerate(sampled_index):
            img_name = Path(str(idx_item[-1] if isinstance(idx_item, tuple) else idx_item)).name
            img_path = (label_dir / img_name).resolve()
            if not img_path.exists():
                continue
            image_paths.append(img_path)
            image_keys.append(f"{stem}/{img_path.name}".lower())
            kept_rows.append(ridx)
        if not image_paths:
            continue
        keep = np.asarray(kept_rows, dtype=np.int64)
        keypoints = keypoints[keep]
        sampled_records.append(
            {
                "joint_names": joint_names,
                "gt_xy": keypoints[:, :, :2],
                "image_keys": image_keys,
                "image_paths": image_paths,
            }
        )
        all_image_paths.extend(image_paths)

    if not sampled_records:
        raise ValueError("No sampled labeled images for ROI classifier.")

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
    pred_lookup = {_index_to_stem_image_key(idx): pred_keypoints[i] for i, idx in enumerate(pred_df.index)}

    bodyparts = [str(bp) for bp in cfg.get("bodyparts", [])]
    fitted_joint_names = bodyparts if bodyparts else sorted({jn.split("|")[-1] for rec in sampled_records for jn in rec["joint_names"]})
    roi_by_joint: dict[str, list[np.ndarray]] = defaultdict(list)
    labels_by_joint: dict[str, list[int]] = defaultdict(list)
    conf_by_joint: dict[str, list[float]] = defaultdict(list)
    xrel_by_joint: dict[str, list[float]] = defaultdict(list)
    yrel_by_joint: dict[str, list[float]] = defaultdict(list)

    for rec in sampled_records:
        joint_names = rec["joint_names"]
        gt_xy = rec["gt_xy"]
        image_keys = rec["image_keys"]
        image_paths = rec["image_paths"]
        pred_for_video = np.full((len(image_keys), len(joint_names), 3), np.nan, dtype=np.float64)
        valid_rows = np.zeros(len(image_keys), dtype=bool)
        for ridx, key in enumerate(image_keys):
            row = pred_lookup.get(key)
            if row is None:
                continue
            valid_rows[ridx] = True
            for j, jn in enumerate(joint_names):
                pidx = pred_joint_idx.get(jn.split("|")[-1])
                if pidx is not None:
                    pred_for_video[ridx, j] = row[pidx]

        for t, img_path in enumerate(image_paths):
            if not valid_rows[t]:
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            all_xy = pred_for_video[t, :, :2]
            good = np.isfinite(all_xy).all(axis=1)
            if np.count_nonzero(good) < 2:
                continue
            pts = all_xy[good]
            min_xy = np.min(pts, axis=0)
            max_xy = np.max(pts, axis=0)
            w = max(8.0, float(max_xy[0] - min_xy[0]))
            h = max(8.0, float(max_xy[1] - min_xy[1]))
            mx = max(8.0, 0.25 * w)
            my = max(8.0, 0.25 * h)
            x0 = float(min_xy[0] - mx)
            y0 = float(min_xy[1] - my)
            x1 = float(max_xy[0] + mx)
            y1 = float(max_xy[1] + my)

            # Crop once per frame.
            roi = _crop_patch_gray(
                img,
                x=(x0 + x1) * 0.5,
                y=(y0 + y1) * 0.5,
                patch_size=int(max(roi_size, np.ceil(max(x1 - x0, y1 - y0)))),
            )
            roi = cv2.resize(roi, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)

            for j, jn in enumerate(joint_names):
                key = jn.split("|")[-1]
                if key not in fitted_joint_names:
                    continue
                x, y, c = pred_for_video[t, j]
                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)):
                    continue
                bw = max(1e-6, x1 - x0)
                bh = max(1e-6, y1 - y0)
                xr = float(np.clip((x - x0) / bw, 0.0, 1.0))
                yr = float(np.clip((y - y0) / bh, 0.0, 1.0))
                label = int(np.isfinite(gt_xy[t, j]).all())
                roi_by_joint[key].append(roi.astype(np.float32))
                labels_by_joint[key].append(label)
                conf_by_joint[key].append(float(np.clip(c, 0.0, 1.0)))
                xrel_by_joint[key].append(xr)
                yrel_by_joint[key].append(yr)

    roi_arr = {k: np.stack(v, axis=0).astype(np.float32) if v else np.empty((0, roi_size, roi_size), dtype=np.float32) for k, v in roi_by_joint.items()}
    label_arr = {k: np.asarray(v, dtype=np.int32) for k, v in labels_by_joint.items()}
    conf_arr = {k: np.asarray(v, dtype=np.float32) for k, v in conf_by_joint.items()}
    xrel_arr = {k: np.asarray(v, dtype=np.float32) for k, v in xrel_by_joint.items()}
    yrel_arr = {k: np.asarray(v, dtype=np.float32) for k, v in yrel_by_joint.items()}
    return fitted_joint_names, roi_arr, label_arr, conf_arr, xrel_arr, yrel_arr


class _TinyRoiVisibilityCNN(nn.Module):
    def __init__(self, roi_size: int, n_joints: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        feat_hw = max(1, roi_size // 8)
        self.fc_img = nn.Linear(64 * feat_hw * feat_hw, 64)
        self.joint_emb = nn.Embedding(n_joints, 8)
        self.head = nn.Sequential(
            nn.Linear(64 + 8 + 3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        roi: torch.Tensor,
        joint_id: torch.Tensor,
        conf: torch.Tensor,
        xrel: torch.Tensor,
        yrel: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(roi).flatten(1)
        x = torch.relu(self.fc_img(x))
        j = self.joint_emb(joint_id)
        meta = torch.stack([conf, xrel, yrel], dim=1)
        z = torch.cat([x, j, meta], dim=1)
        return self.head(z).squeeze(1)


def _fit_from_roi_cnn(
    project_path: str | Path,
    shuffle: int,
    score_model_name: str,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
    roi_size: int = 96,
    epochs: int = 8,
    batch_size: int = 192,
    learning_rate: float = 1e-3,
    torch_device: str | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for ROI CNN visibility classifier.")

    data = _collect_fit_data(
        project_path=project_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
    )
    fitted_joint_names, roi_by_joint, labels_by_joint, conf_by_joint, xrel_by_joint, yrel_by_joint = _collect_roi_samples(
        project_path=project_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        use_cached_predictions=use_cached_predictions,
        roi_size=roi_size,
    )

    split = _build_joint_split(
        labels_by_joint=labels_by_joint,
        joint_keys=fitted_joint_names,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )
    joint_to_idx = {k: i for i, k in enumerate(fitted_joint_names)}
    r_train: list[np.ndarray] = []
    y_train: list[np.ndarray] = []
    j_train: list[np.ndarray] = []
    c_train: list[np.ndarray] = []
    xr_train: list[np.ndarray] = []
    yr_train: list[np.ndarray] = []

    for key in fitted_joint_names:
        r = roi_by_joint.get(key, np.empty((0, roi_size, roi_size), dtype=np.float32))
        y = labels_by_joint.get(key, np.empty((0,), dtype=np.int32))
        c = conf_by_joint.get(key, np.empty((0,), dtype=np.float32))
        xr = xrel_by_joint.get(key, np.empty((0,), dtype=np.float32))
        yr = yrel_by_joint.get(key, np.empty((0,), dtype=np.float32))
        n = min(len(r), len(y), len(c), len(xr), len(yr))
        if n == 0:
            continue
        train_idx, _ = split.get(key, (np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)))
        train_idx = train_idx[train_idx < n]
        if train_idx.size == 0:
            continue
        r_train.append(r[train_idx])
        y_train.append(y[train_idx])
        c_train.append(c[train_idx])
        xr_train.append(xr[train_idx])
        yr_train.append(yr[train_idx])
        j_train.append(np.full(train_idx.size, joint_to_idx[key], dtype=np.int64))

    if not r_train:
        raise ValueError("No train samples for ROI CNN.")

    torch.manual_seed(int(test_seed))
    np.random.seed(int(test_seed))
    r_train_arr = np.concatenate(r_train, axis=0)[:, None, :, :]
    y_train_arr = np.concatenate(y_train, axis=0).astype(np.float32)
    j_train_arr = np.concatenate(j_train, axis=0).astype(np.int64)
    c_train_arr = np.concatenate(c_train, axis=0).astype(np.float32)
    xr_train_arr = np.concatenate(xr_train, axis=0).astype(np.float32)
    yr_train_arr = np.concatenate(yr_train, axis=0).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(r_train_arr),
        torch.from_numpy(j_train_arr),
        torch.from_numpy(c_train_arr),
        torch.from_numpy(xr_train_arr),
        torch.from_numpy(yr_train_arr),
        torch.from_numpy(y_train_arr),
    )
    device = _resolve_torch_device(torch_device)
    loader = DataLoader(
        dataset,
        batch_size=int(max(16, batch_size)),
        shuffle=True,
        num_workers=0,
        pin_memory=bool(device and device.type == "cuda"),
    )
    model = _TinyRoiVisibilityCNN(roi_size=roi_size, n_joints=len(fitted_joint_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    pos = float(np.sum(y_train_arr == 1))
    neg = float(np.sum(y_train_arr == 0))
    pos_weight = torch.tensor([max(1e-6, neg / max(pos, 1e-6))], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for _ in range(int(max(1, epochs))):
        for rb, jb, cb, xb, yb, lb in loader:
            rb = rb.to(device, non_blocking=True)
            jb = jb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(rb, jb, cb, xb, yb)
            loss = criterion(logits, lb)
            loss.backward()
            optimizer.step()

    model.eval()
    per_joint_scores_fit: dict[str, np.ndarray] = {}
    labels_fit_by_joint: dict[str, np.ndarray] = {}
    per_joint_scores_eval: dict[str, np.ndarray] = {}
    labels_eval_by_joint: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for key in fitted_joint_names:
            r = roi_by_joint.get(key, np.empty((0, roi_size, roi_size), dtype=np.float32))
            y = labels_by_joint.get(key, np.empty((0,), dtype=np.int32))
            c = conf_by_joint.get(key, np.empty((0,), dtype=np.float32))
            xr = xrel_by_joint.get(key, np.empty((0,), dtype=np.float32))
            yr = yrel_by_joint.get(key, np.empty((0,), dtype=np.float32))
            n = min(len(r), len(y), len(c), len(xr), len(yr))
            r = r[:n]
            y = y[:n]
            c = c[:n]
            xr = xr[:n]
            yr = yr[:n]
            train_idx, test_idx = split.get(key, (np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)))
            train_idx = train_idx[train_idx < n]
            test_idx = test_idx[test_idx < n]
            if n > 0:
                logits = model(
                    torch.from_numpy(r[:, None, :, :]).to(device, non_blocking=True),
                    torch.full((n,), joint_to_idx[key], dtype=torch.long, device=device),
                    torch.from_numpy(c.astype(np.float32)).to(device, non_blocking=True),
                    torch.from_numpy(xr.astype(np.float32)).to(device, non_blocking=True),
                    torch.from_numpy(yr.astype(np.float32)).to(device, non_blocking=True),
                )
                probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
            else:
                probs = np.array([], dtype=np.float64)
            per_joint_scores_fit[key] = probs[train_idx]
            labels_fit_by_joint[key] = y[train_idx]
            per_joint_scores_eval[key] = probs[test_idx]
            labels_eval_by_joint[key] = y[test_idx]

    return _finalize_fit(
        data=data,
        per_joint_scores=per_joint_scores_fit,
        project_path=project_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        score_model_name=score_model_name,
        labels_fit_by_joint=labels_fit_by_joint,
        scores_eval_by_joint=per_joint_scores_eval,
        labels_eval_by_joint=labels_eval_by_joint,
        cutoff_split="test",
    )


def _collect_roi_context_samples(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    use_cached_predictions: bool = True,
    roi_size: int = 128,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    project_root = Path(project_path).resolve()
    config_path = project_root / "config.yaml"
    cfg = deeplabcut.auxiliaryfunctions.read_config(str(config_path))
    video_sets = cfg.get("video_sets", {})
    scorer = str(cfg.get("scorer", "")).strip()
    labeled_root = project_root / "labeled-data"
    cache_dir = project_root / ".kalman_fit_cache" / f"shuffle{shuffle}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    bodyparts = [str(bp) for bp in cfg.get("bodyparts", [])]
    fitted_joint_names = bodyparts
    if not fitted_joint_names:
        raise ValueError("config bodyparts is empty; cannot build context model.")
    joint_to_idx = {k: i for i, k in enumerate(fitted_joint_names)}
    jcount = len(fitted_joint_names)

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
        keypoints = keypoints[sample_idx]
        sampled_index = np.array(df.index, dtype=object)[valid_frame_idx][order][sample_idx]

        image_paths: list[Path] = []
        image_keys: list[str] = []
        kept_rows: list[int] = []
        for ridx, idx_item in enumerate(sampled_index):
            img_name = Path(str(idx_item[-1] if isinstance(idx_item, tuple) else idx_item)).name
            img_path = (label_dir / img_name).resolve()
            if not img_path.exists():
                continue
            image_paths.append(img_path)
            image_keys.append(f"{stem}/{img_path.name}".lower())
            kept_rows.append(ridx)
        if not image_paths:
            continue
        keep = np.asarray(kept_rows, dtype=np.int64)
        keypoints = keypoints[keep]
        sampled_records.append(
            {
                "joint_names": joint_names,
                "gt_xy": keypoints[:, :, :2],
                "image_keys": image_keys,
                "image_paths": image_paths,
            }
        )
        all_image_paths.extend(image_paths)

    if not sampled_records:
        raise ValueError("No sampled frames for ROI context classifier.")

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
    pred_lookup = {_index_to_stem_image_key(idx): pred_keypoints[i] for i, idx in enumerate(pred_df.index)}

    rois: list[np.ndarray] = []
    metas: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for rec in sampled_records:
        joint_names = rec["joint_names"]
        gt_xy = rec["gt_xy"]
        image_keys = rec["image_keys"]
        image_paths = rec["image_paths"]
        pred_for_video = np.full((len(image_keys), len(joint_names), 3), np.nan, dtype=np.float64)
        valid_rows = np.zeros(len(image_keys), dtype=bool)
        for ridx, key in enumerate(image_keys):
            row = pred_lookup.get(key)
            if row is None:
                continue
            valid_rows[ridx] = True
            for j, jn in enumerate(joint_names):
                pidx = pred_joint_idx.get(jn.split("|")[-1])
                if pidx is not None:
                    pred_for_video[ridx, j] = row[pidx]

        for t, img_path in enumerate(image_paths):
            if not valid_rows[t]:
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            xy_all = np.full((jcount, 2), np.nan, dtype=np.float32)
            conf_all = np.zeros((jcount,), dtype=np.float32)
            label_all = np.zeros((jcount,), dtype=np.float32)
            for j, jn in enumerate(joint_names):
                key = jn.split("|")[-1]
                if key not in joint_to_idx:
                    continue
                idx = joint_to_idx[key]
                x, y, c = pred_for_video[t, j]
                if np.isfinite(x) and np.isfinite(y):
                    xy_all[idx] = [x, y]
                if np.isfinite(c):
                    conf_all[idx] = float(np.clip(c, 0.0, 1.0))
                label_all[idx] = float(int(np.isfinite(gt_xy[t, j]).all()))

            good = np.isfinite(xy_all).all(axis=1)
            if np.count_nonzero(good) < 2:
                continue
            pts = xy_all[good]
            min_xy = np.min(pts, axis=0)
            max_xy = np.max(pts, axis=0)
            w = max(8.0, float(max_xy[0] - min_xy[0]))
            h = max(8.0, float(max_xy[1] - min_xy[1]))
            mx = max(8.0, 0.25 * w)
            my = max(8.0, 0.25 * h)
            x0 = float(min_xy[0] - mx)
            y0 = float(min_xy[1] - my)
            x1 = float(max_xy[0] + mx)
            y1 = float(max_xy[1] + my)
            roi = _crop_patch_gray(
                img,
                x=(x0 + x1) * 0.5,
                y=(y0 + y1) * 0.5,
                patch_size=int(max(roi_size, np.ceil(max(x1 - x0, y1 - y0)))),
            )
            roi = cv2.resize(roi, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)

            bw = max(1e-6, x1 - x0)
            bh = max(1e-6, y1 - y0)
            xrel = np.where(np.isfinite(xy_all[:, 0]), np.clip((xy_all[:, 0] - x0) / bw, 0.0, 1.0), 0.5)
            yrel = np.where(np.isfinite(xy_all[:, 1]), np.clip((xy_all[:, 1] - y0) / bh, 0.0, 1.0), 0.5)
            meta = np.concatenate([xrel.astype(np.float32), yrel.astype(np.float32), conf_all.astype(np.float32)], axis=0)

            rois.append(roi)
            metas.append(meta)
            labels.append(label_all.astype(np.float32))

    if not rois:
        raise ValueError("No valid ROI context samples.")
    roi_arr = np.stack(rois, axis=0).astype(np.float32)
    meta_arr = np.stack(metas, axis=0).astype(np.float32)
    label_arr = np.stack(labels, axis=0).astype(np.float32)
    return fitted_joint_names, roi_arr, meta_arr, label_arr


class _TinyRoiContextCNN(nn.Module):
    def __init__(self, roi_size: int, meta_dim: int, n_joints: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        feat_hw = max(1, roi_size // 8)
        self.img_fc = nn.Sequential(
            nn.Linear(64 * feat_hw * feat_hw, 128),
            nn.ReLU(inplace=True),
        )
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_joints),
        )

    def forward(self, roi: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        x = self.conv(roi).flatten(1)
        x = self.img_fc(x)
        m = self.meta_fc(meta)
        return self.head(torch.cat([x, m], dim=1))


def _fit_from_roi_context_cnn(
    project_path: str | Path,
    shuffle: int,
    score_model_name: str,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
    roi_size: int = 128,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 6e-4,
    torch_device: str | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for ROI context CNN.")

    data = _collect_fit_data(
        project_path=project_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
    )
    joint_names, roi_arr, meta_arr, label_arr = _collect_roi_context_samples(
        project_path=project_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        use_cached_predictions=use_cached_predictions,
        roi_size=roi_size,
    )
    n_samples = roi_arr.shape[0]
    n_joints = label_arr.shape[1]
    train_idx, test_idx = _build_frame_split(
        n_samples=n_samples,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )

    device = _resolve_torch_device(torch_device)
    torch.manual_seed(int(test_seed))
    np.random.seed(int(test_seed))
    train_dataset = TensorDataset(
        torch.from_numpy(roi_arr[train_idx][:, None, :, :]),
        torch.from_numpy(meta_arr[train_idx]),
        torch.from_numpy(label_arr[train_idx]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(max(16, batch_size)),
        shuffle=True,
        num_workers=0,
        pin_memory=bool(device and device.type == "cuda"),
    )

    model = _TinyRoiContextCNN(roi_size=roi_size, meta_dim=meta_arr.shape[1], n_joints=n_joints).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    pos = np.sum(label_arr[train_idx] == 1, axis=0).astype(np.float32)
    neg = np.sum(label_arr[train_idx] == 0, axis=0).astype(np.float32)
    pos_weight = torch.from_numpy(np.maximum(1e-6, neg / np.maximum(pos, 1e-6))).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for _ in range(int(max(1, epochs))):
        for rb, mb, yb in train_loader:
            rb = rb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(rb, mb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_all = model(
            torch.from_numpy(roi_arr[:, None, :, :]).to(device, non_blocking=True),
            torch.from_numpy(meta_arr).to(device, non_blocking=True),
        )
        probs_all = torch.sigmoid(logits_all).cpu().numpy().astype(np.float64)

    per_joint_scores_fit: dict[str, np.ndarray] = {}
    labels_fit_by_joint: dict[str, np.ndarray] = {}
    per_joint_scores_eval: dict[str, np.ndarray] = {}
    labels_eval_by_joint: dict[str, np.ndarray] = {}
    for j, key in enumerate(joint_names):
        per_joint_scores_fit[key] = probs_all[train_idx, j]
        labels_fit_by_joint[key] = label_arr[train_idx, j].astype(np.int32)
        per_joint_scores_eval[key] = probs_all[test_idx, j]
        labels_eval_by_joint[key] = label_arr[test_idx, j].astype(np.int32)

    return _finalize_fit(
        data=data,
        per_joint_scores=per_joint_scores_fit,
        project_path=project_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        score_model_name=score_model_name,
        labels_fit_by_joint=labels_fit_by_joint,
        scores_eval_by_joint=per_joint_scores_eval,
        labels_eval_by_joint=labels_eval_by_joint,
        cutoff_split="test",
    )


def _score_builder1(raw: dict[str, np.ndarray]) -> np.ndarray:
    return _clip01(raw["likelihood"])


def _score_builder2(raw: dict[str, np.ndarray]) -> np.ndarray:
    curr = _clip01(raw["likelihood"])
    prev = _clip01(raw["prev"])
    nxt = _clip01(raw["next"])
    with np.errstate(invalid="ignore"):
        nei = np.nanmean(np.vstack([prev, nxt]), axis=0)
    nei = np.where(np.isfinite(nei), nei, curr)
    score = np.sqrt(np.maximum(curr * np.maximum(nei, 1e-6), 0.0))
    return _clip01(score)


def _score_builder3(raw: dict[str, np.ndarray]) -> np.ndarray:
    curr = _clip01(raw["likelihood"])
    bone = np.clip(np.nan_to_num(raw["bone_med"], nan=0.9, posinf=0.9, neginf=0.9), 0.0, 3.0)
    neigh = _clip01(raw["neighbor_conf"])
    rel = _clip01(raw["rel_conf"])
    score = (curr ** 1.1) * (np.exp(-1.8 * bone)) * np.power(np.maximum(neigh, 1e-6), 0.45)
    score = score * (0.82 + 0.18 * rel)
    return _clip01(score)


def _make_rf_model() -> Any:
    if RandomForestClassifier is None:
        raise ImportError("scikit-learn is required for kalman_fitter4.")
    return RandomForestClassifier(
        n_estimators=260,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def _make_hgb_model() -> Any:
    if HistGradientBoostingClassifier is None:
        raise ImportError("scikit-learn is required for kalman_fitter5.")
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.06,
        max_iter=180,
        min_samples_leaf=18,
        random_state=42,
    )


def _make_lr_model() -> Any:
    if LogisticRegression is None:
        raise ImportError("scikit-learn is required for kalman_fitter6.")
    return LogisticRegression(
        max_iter=300,
        class_weight="balanced",
    )


def _make_linear_svc_model() -> Any:
    if LinearSVC is None:
        raise ImportError("scikit-learn is required for kalman_fitter7.")
    return LinearSVC(
        C=1.0,
        class_weight="balanced",
        dual=False,
        max_iter=5000,
    )


def _make_extra_trees_model() -> Any:
    if ExtraTreesClassifier is None:
        raise ImportError("scikit-learn is required for kalman_fitter8.")
    return ExtraTreesClassifier(
        n_estimators=340,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def _make_gbrt_model() -> Any:
    if GradientBoostingClassifier is None:
        raise ImportError("scikit-learn is required for kalman_fitter9.")
    return GradientBoostingClassifier(
        n_estimators=280,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )


def _make_knn_model() -> Any:
    if KNeighborsClassifier is None:
        raise ImportError("scikit-learn is required for kalman_fitter10.")
    return KNeighborsClassifier(
        n_neighbors=35,
        weights="distance",
        metric="minkowski",
        p=2,
    )

# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.1538, test global vis=0.9528, mean joint invis=0.1484, joints>=0.8=0/19.
def kalman_fitter1(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_score_builder(
        project_path=project_path,
        shuffle=shuffle,
        score_builder=_score_builder1,
        score_model_name="likelihood_raw",
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )


# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.1522, test global vis=0.9528, mean joint invis=0.1397, joints>=0.8=0/19.
def kalman_fitter2(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_score_builder(
        project_path=project_path,
        shuffle=shuffle,
        score_builder=_score_builder2,
        score_model_name="likelihood_temporal_geom",
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )


# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.2848, test global vis=0.9528, mean joint invis=0.3372, joints>=0.8=0/19.
def kalman_fitter3(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_score_builder(
        project_path=project_path,
        shuffle=shuffle,
        score_builder=_score_builder3,
        score_model_name="likelihood_bone_penalty",
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )


# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.3209, test global vis=0.9528, mean joint invis=0.3507, joints>=0.8=0/19.
def kalman_fitter4(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_ml_model(
        project_path=project_path,
        shuffle=shuffle,
        score_model_name="rf_per_joint_dispersion",
        model_factory=_make_rf_model,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )


# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.2554, test global vis=0.9528, mean joint invis=0.3058, joints>=0.8=0/19.
def kalman_fitter5(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_ml_model(
        project_path=project_path,
        shuffle=shuffle,
        score_model_name="hgb_per_joint_dispersion",
        model_factory=_make_hgb_model,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )


# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.3277, test global vis=0.9528, mean joint invis=0.3802, joints>=0.8=0/19.
def kalman_fitter6(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_ml_model(
        project_path=project_path,
        shuffle=shuffle,
        score_model_name="logreg_per_joint_dispersion",
        model_factory=_make_lr_model,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )


# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.3189, test global vis=0.9528, mean joint invis=0.4001, joints>=0.8=1/19.
def kalman_fitter7(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_ml_model(
        project_path=project_path,
        shuffle=shuffle,
        score_model_name="linearsvc_per_joint_dispersion",
        model_factory=_make_linear_svc_model,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )


# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.3117, test global vis=0.9528, mean joint invis=0.3507, joints>=0.8=0/19.
def kalman_fitter8(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_ml_model(
        project_path=project_path,
        shuffle=shuffle,
        score_model_name="extratrees_per_joint_dispersion",
        model_factory=_make_extra_trees_model,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )


# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.2405, test global vis=0.9528, mean joint invis=0.3188, joints>=0.8=0/19.
def kalman_fitter9(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_ml_model(
        project_path=project_path,
        shuffle=shuffle,
        score_model_name="gbrt_per_joint_dispersion",
        model_factory=_make_gbrt_model,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )


# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.2110, test global vis=0.9537, mean joint invis=0.3375, joints>=0.8=0/19.
def kalman_fitter10(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_ml_model(
        project_path=project_path,
        shuffle=shuffle,
        score_model_name="knn_per_joint_dispersion",
        model_factory=_make_knn_model,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )


# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.3091, test global vis=0.9528, mean joint invis=0.3440, joints>=0.8=0/19.
def kalman_fitter11(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_ml_ensemble(
        project_path=project_path,
        shuffle=shuffle,
        score_model_name="rf_hgb_weighted_ensemble",
        model_factories=[_make_rf_model, _make_hgb_model],
        model_weights=[0.65, 0.35],
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
    )


# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.3287, test global vis=0.9528, mean joint invis=0.3078, joints>=0.8=0/19.
def kalman_fitter12(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
    patch_size: int = 48,
    cnn_epochs: int = 6,
    cnn_batch_size: int = 256,
    cnn_learning_rate: float = 1e-3,
    torch_device: str | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_patch_cnn(
        project_path=project_path,
        shuffle=shuffle,
        score_model_name="tiny_patch_cnn",
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
        patch_size=patch_size,
        epochs=cnn_epochs,
        batch_size=cnn_batch_size,
        learning_rate=cnn_learning_rate,
        torch_device=torch_device,
    )


# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.3684, test global vis=0.9528, mean joint invis=0.3799, joints>=0.8=0/19.
def kalman_fitter13(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
    roi_size: int = 96,
    cnn_epochs: int = 8,
    cnn_batch_size: int = 192,
    cnn_learning_rate: float = 1e-3,
    torch_device: str | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_roi_cnn(
        project_path=project_path,
        shuffle=shuffle,
        score_model_name="tiny_roi_cnn_v1",
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
        roi_size=roi_size,
        epochs=cnn_epochs,
        batch_size=cnn_batch_size,
        learning_rate=cnn_learning_rate,
        torch_device=torch_device,
    )


# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.3989, test global vis=0.9528, mean joint invis=0.4246, joints>=0.8=0/19.
def kalman_fitter14(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
    roi_size: int = 128,
    cnn_epochs: int = 12,
    cnn_batch_size: int = 128,
    cnn_learning_rate: float = 8e-4,
    torch_device: str | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_roi_cnn(
        project_path=project_path,
        shuffle=shuffle,
        score_model_name="tiny_roi_cnn_v2",
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
        roi_size=roi_size,
        epochs=cnn_epochs,
        batch_size=cnn_batch_size,
        learning_rate=cnn_learning_rate,
        torch_device=torch_device,
    )


# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.4675, test global vis=0.9528, mean joint invis=0.4778, joints>=0.8=0/19.
def kalman_fitter15(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
    roi_size: int = 128,
    cnn_epochs: int = 20,
    cnn_batch_size: int = 96,
    cnn_learning_rate: float = 6e-4,
    torch_device: str | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_roi_cnn(
        project_path=project_path,
        shuffle=shuffle,
        score_model_name="tiny_roi_cnn_v3",
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
        roi_size=roi_size,
        epochs=cnn_epochs,
        batch_size=cnn_batch_size,
        learning_rate=cnn_learning_rate,
        torch_device=torch_device,
    )


# 2026-03-08 holdout test (projects/rat, shuffle=3, target=0.95, test_fraction=0.2, seed=42):
# test global invis=0.4360, test global vis=0.9528, mean joint invis=0.4786, joints>=0.8=0/19.
def kalman_fitter16(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
    roi_size: int = 160,
    cnn_epochs: int = 24,
    cnn_batch_size: int = 64,
    cnn_learning_rate: float = 5e-4,
    torch_device: str | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_roi_cnn(
        project_path=project_path,
        shuffle=shuffle,
        score_model_name="tiny_roi_cnn_v4",
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
        roi_size=roi_size,
        epochs=cnn_epochs,
        batch_size=cnn_batch_size,
        learning_rate=cnn_learning_rate,
        torch_device=torch_device,
    )


# TODO: update metrics after test.
def kalman_fitter17(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
    roi_size: int = 128,
    cnn_epochs: int = 20,
    cnn_batch_size: int = 128,
    cnn_learning_rate: float = 6e-4,
    torch_device: str | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_roi_context_cnn(
        project_path=project_path,
        shuffle=shuffle,
        score_model_name="tiny_roi_context_cnn_v1",
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
        roi_size=roi_size,
        epochs=cnn_epochs,
        batch_size=cnn_batch_size,
        learning_rate=cnn_learning_rate,
        torch_device=torch_device,
    )


# TODO: update metrics after test.
def kalman_fitter18(
    project_path: str | Path,
    shuffle: int,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    max_labeled_frames_per_video: int = 1500,
    max_speed_samples_per_joint: int = 30000,
    use_cached_predictions: bool = True,
    dispersion_visible_recall_target: float = 0.95,
    test_fraction: float = 0.2,
    test_size: int | None = None,
    test_seed: int = 42,
    roi_size: int = 160,
    cnn_epochs: int = 28,
    cnn_batch_size: int = 96,
    cnn_learning_rate: float = 4e-4,
    torch_device: str | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    return _fit_from_roi_context_cnn(
        project_path=project_path,
        shuffle=shuffle,
        score_model_name="tiny_roi_context_cnn_v2",
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        max_labeled_frames_per_video=max_labeled_frames_per_video,
        max_speed_samples_per_joint=max_speed_samples_per_joint,
        use_cached_predictions=use_cached_predictions,
        dispersion_visible_recall_target=dispersion_visible_recall_target,
        test_fraction=test_fraction,
        test_size=test_size,
        test_seed=test_seed,
        roi_size=roi_size,
        epochs=cnn_epochs,
        batch_size=cnn_batch_size,
        learning_rate=cnn_learning_rate,
        torch_device=torch_device,
    )
