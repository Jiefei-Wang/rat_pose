"""
RMSE evaluation utilities for DeepLabCut models.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from deeplabcut.core.metrics.api import prepare_evaluation_data
from deeplabcut.core.metrics.distance_metrics import match_predictions_for_rmse
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.pose_estimation_pytorch import utils
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    get_inference_runners,
    get_model_snapshots,
)
from deeplabcut.pose_estimation_pytorch.data import DLCLoader
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.utils import auxiliaryfunctions
from modules.label_csv_utils import get_image_names, load_keypoints


def compute_rmse_no_cutoff(
    config: str | Path,
    test_labels_root: str | Path,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    snapshotindex: int | str | None = None,
    device: str | None = None,
    comparison_bodyparts: str | list[str] | None = None,
    video_names: str | list[str] | None = None,
    detector_snapshot_index: int | None = None,
    pretty: bool = False,
) -> dict:
    """Compute total and per-keypoint RMSE on external test images without pcutoff filtering."""
    config = str(config)
    cfg = auxiliaryfunctions.read_config(config)
    test_labels_root = Path(test_labels_root)

    if snapshotindex is None:
        snapshotindex = cfg.get("snapshotindex", -1)
    if detector_snapshot_index is None:
        detector_snapshot_index = cfg.get("detector_snapshotindex", None)

    loader = DLCLoader(config=config, shuffle=shuffle, trainset_index=trainingsetindex)
    if device is not None:
        loader.model_cfg["device"] = device
    loader.model_cfg["device"] = utils.resolve_device(loader.model_cfg)

    snapshots = get_model_snapshots(snapshotindex, model_folder=loader.model_folder, task=loader.pose_task)
    if not snapshots:
        raise FileNotFoundError(f"No snapshots in {loader.model_folder} for index {snapshotindex}")
    snapshot = snapshots[-1]

    detector_path = None
    if loader.pose_task == Task.TOP_DOWN and detector_snapshot_index is not None:
        det_snaps = get_model_snapshots(detector_snapshot_index, loader.model_folder, Task.DETECT)
        if det_snaps:
            detector_path = det_snaps[-1].path

    parameters = loader.get_dataset_parameters()
    pose_runner, detector_runner = get_inference_runners(
        model_config=loader.model_cfg,
        snapshot_path=snapshot.path,
        max_individuals=parameters.max_num_animals,
        num_bodyparts=parameters.num_joints,
        num_unique_bodyparts=parameters.num_unique_bpts,
        with_identity=loader.model_cfg["metadata"]["with_identity"],
        detector_path=detector_path,
    )

    project_bodyparts = list(parameters.bodyparts)
    single_animal = parameters.max_num_animals == 1

    conversion_array = None
    if wi_cfg := loader.model_cfg["train_settings"].get("weight_init"):
        wi = WeightInitialization.from_dict(wi_cfg)
        if wi.memory_replay:
            conversion_array = wi.conversion_array

    video_dirs = _resolve_video_dirs(test_labels_root, cfg, video_names)
    if not video_dirs:
        raise FileNotFoundError(f"No sub-folders found under {test_labels_root}")

    all_pixel_errors = []
    per_video_rows = []

    for vdir in video_dirs:
        gt_df, gt_bodyparts = _load_gt_dataframe(vdir)
        if gt_df is None:
            print(f"[skip] no label file in {vdir.name}")
            continue

        image_paths = []
        gt_arrays = []

        for img_name, row in gt_df.iterrows():
            img_path = vdir / img_name
            if not img_path.exists():
                print(f"[skip] image not found: {img_path}")
                continue

            coords = row.values.astype(float)
            n_bpts = len(coords) // 2
            xy = coords.reshape(n_bpts, 2)
            vis = np.where(np.isnan(xy).any(axis=1), 0.0, 2.0)
            gt_kpts = np.concatenate([xy, vis[:, None]], axis=1)[np.newaxis, ...]

            image_paths.append(str(img_path))
            gt_arrays.append(gt_kpts)

        if not image_paths:
            print(f"[skip] no images found in {vdir.name}")
            continue

        images_input = image_paths
        if detector_runner is not None:
            bbox_preds = detector_runner.inference(images=tqdm(image_paths, desc=f"detect {vdir.name}", leave=False))
            images_input = list(zip(image_paths, bbox_preds))

        raw_preds = pose_runner.inference(images=tqdm(images_input, desc=f"pose {vdir.name}", leave=False))

        gt_pose = {}
        pred_pose = {}
        for img_path, gt_arr, pred_dict in zip(image_paths, gt_arrays, raw_preds):
            pred_bpts = pred_dict["bodyparts"]
            if conversion_array is not None:
                pred_bpts = pred_bpts[:, conversion_array]

            gt_pose[img_path] = _align_bodyparts(gt_arr, gt_bodyparts, project_bodyparts)
            pred_pose[img_path] = pred_bpts

        eval_bodyparts = list(project_bodyparts)
        kpt_idx = _get_keypoints_to_use(eval_bodyparts, comparison_bodyparts)
        if kpt_idx is not None:
            gt_pose = {k: v[:, kpt_idx] for k, v in gt_pose.items()}
            pred_pose = {k: v[:, kpt_idx] for k, v in pred_pose.items()}
            eval_bodyparts = [eval_bodyparts[i] for i in kpt_idx]

        data = prepare_evaluation_data(gt_pose, pred_pose)
        matches = match_predictions_for_rmse(data, single_animal)

        if matches:
            pe = np.stack([m.pixel_errors() for m in matches])
            all_pixel_errors.append(pe)
            valid = ~np.isnan(pe)
            per_video_rows.append(
                {
                    "video": vdir.name,
                    "mse": float(np.nanmean(pe**2)),
                    "rmse": float(np.sqrt(np.nanmean(pe**2))),
                    "mean_euclidean": float(np.nanmean(pe)),
                    "support": int(np.sum(valid)),
                }
            )
        else:
            per_video_rows.append(
                {
                    "video": vdir.name,
                    "mse": float("nan"),
                    "rmse": float("nan"),
                    "mean_euclidean": float("nan"),
                    "support": 0,
                }
            )

    if not all_pixel_errors:
        raise RuntimeError("No valid matches across any test video.")

    pixel_errors = np.concatenate(all_pixel_errors, axis=0)
    total_support = int(np.sum(~np.isnan(pixel_errors)))
    total_mse = float(np.nanmean(pixel_errors**2))
    total_rmse = float(np.sqrt(np.nanmean(pixel_errors**2)))
    total_mean_euc = float(np.nanmean(pixel_errors))

    kpt_rows = []
    for idx, bpt in enumerate(eval_bodyparts):
        col = pixel_errors[:, idx]
        support = int(np.sum(~np.isnan(col)))
        if support > 0:
            mse = float(np.nanmean(col**2))
            rmse = float(np.sqrt(np.nanmean(col**2)))
            mean_euclidean = float(np.nanmean(col))
        else:
            mse = float("nan")
            rmse = float("nan")
            mean_euclidean = float("nan")
        kpt_rows.append(
            {
                "bodypart": bpt,
                "mse": mse,
                "rmse": rmse,
                "mean_euclidean": mean_euclidean,
                "support": support,
            }
        )

    result = {
        "total_mse": total_mse,
        "total_rmse": total_rmse,
        "total_mean_euclidean": total_mean_euc,
        "total_support": total_support,
        "per_keypoint": pd.DataFrame(kpt_rows),
        "per_video": pd.DataFrame(per_video_rows),
    }

    if pretty:
        print_rmse_report(result)

    return result


def _resolve_video_dirs(
    test_labels_root: Path,
    cfg: dict,
    video_names: str | list[str] | None = None,
) -> list[Path]:
    if (test_labels_root / "config.yaml").exists():
        labeled_data_root = test_labels_root / "labeled-data"
        return _find_labeled_video_dirs(labeled_data_root, video_names)

    if (test_labels_root / "labeled-data").is_dir():
        labeled_data_root = test_labels_root / "labeled-data"
        return _find_labeled_video_dirs(labeled_data_root, video_names)

    return _find_labeled_video_dirs(test_labels_root, video_names)


def _find_labeled_video_dirs(root: Path, video_names: str | list[str] | None = None) -> list[Path]:
    if isinstance(video_names, str):
        video_names = [video_names]
    allowed = set(video_names) if video_names is not None else None

    video_dirs = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if allowed is not None and folder.name not in allowed:
            continue
        has_csv = any(folder.glob("CollectedData_*.csv"))
        has_png = any(folder.glob("*.png"))
        if has_csv and has_png:
            video_dirs.append(folder)
    return video_dirs


def _load_gt_dataframe(folder: Path) -> tuple[pd.DataFrame | None, list[str]]:
    h5_files = sorted(folder.glob("CollectedData_*.h5"))
    if h5_files:
        return _flatten_h5_labels(pd.read_hdf(h5_files[0]))

    csv_files = sorted(folder.glob("CollectedData_*.csv"))
    if csv_files:
        df = load_keypoints(csv_files[0])
        if df.empty:
            return None, []
        image_names = get_image_names(df)
        flat_df = df.drop(columns=["frame"]).copy()
        flat_df.index = image_names
        bodyparts = [col[:-2] for col in flat_df.columns if col.endswith("_x")]
        return flat_df, bodyparts

    return None, []


def _flatten_h5_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    bodyparts = []
    flat = {}
    for col in df.columns:
        if len(col) < 3:
            continue
        bodypart = str(col[1]).strip()
        coord = str(col[2]).strip().lower()
        if coord not in ("x", "y"):
            continue
        flat[f"{bodypart}_{coord}"] = pd.to_numeric(df[col], errors="coerce")
        if bodypart not in bodyparts:
            bodyparts.append(bodypart)

    flat_df = pd.DataFrame(flat)
    flat_df.index = [idx[-1] if isinstance(idx, tuple) else idx for idx in df.index]
    return flat_df, bodyparts


def _align_bodyparts(gt: np.ndarray, gt_bodyparts: list[str], target_bodyparts: list[str]) -> np.ndarray:
    n_idv = gt.shape[0]
    out = np.full((n_idv, len(target_bodyparts), 3), np.nan)
    bp_to_idx = {bp: i for i, bp in enumerate(gt_bodyparts)}
    for tgt_i, bp in enumerate(target_bodyparts):
        if bp in bp_to_idx:
            out[:, tgt_i, :] = gt[:, bp_to_idx[bp], :]
    return out


def _get_keypoints_to_use(
    bodyparts: list[str],
    comparison_bodyparts: str | list[str] | None,
) -> list[int] | None:
    if comparison_bodyparts is None or comparison_bodyparts == "all":
        return None
    if isinstance(comparison_bodyparts, str):
        comparison_bodyparts = [comparison_bodyparts]
    return [bodyparts.index(bp) for bp in comparison_bodyparts if bp in bodyparts]


def print_rmse_report(result: dict) -> None:
    print("=" * 60)
    print("RMSE Evaluation Report (no pcutoff filtering)")
    print("=" * 60)
    print(f"  Total MSE:                         {result['total_mse']:.4f} px^2")
    print(f"  Total RMSE (root-mean-square):   {result['total_rmse']:.4f} px")
    print(f"  Total Mean Euclidean Distance:    {result['total_mean_euclidean']:.4f} px")
    print(f"  Total valid keypoint evaluations: {result['total_support']}")
    print("-" * 60)
    print("Per-keypoint breakdown:")
    print(result["per_keypoint"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if "per_video" in result:
        print("-" * 60)
        print("Per-video breakdown:")
        print(result["per_video"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 60)
