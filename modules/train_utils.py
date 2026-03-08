
from pathlib import Path
import re
import shutil
import yaml


def get_shuffle_train_dir(project_path, shuffle, iteration=0):
    base_dir = Path(project_path) / "dlc-models-pytorch" / f"iteration-{iteration}"
    candidates = sorted(base_dir.glob(f"*shuffle{shuffle}/train"))
    if not candidates:
        raise FileNotFoundError(f"No training directory found for shuffle={shuffle} under {base_dir}")
    if len(candidates) > 1:
        print(f"Multiple candidates found for shuffle {shuffle}; using: {candidates[0]}")
    return candidates[0]


def _pick_best_snapshot(train_dir, prefix):
    exact = train_dir / f"{prefix}.pt"
    if exact.exists():
        return exact

    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)\.pt$")
    numbered = []
    for ckpt in train_dir.glob(f"{prefix}-*.pt"):
        match = pattern.match(ckpt.name)
        if match:
            numbered.append((int(match.group(1)), ckpt))

    if not numbered:
        raise FileNotFoundError(f"No checkpoint found for prefix '{prefix}' in {train_dir}")

    return max(numbered, key=lambda x: x[0])[1]


def resolve_best_transfer_paths(project_path, source_shuffle, iteration=0):
    source_train_dir = get_shuffle_train_dir(project_path, shuffle=source_shuffle, iteration=iteration)
    snapshot_path = _pick_best_snapshot(source_train_dir, "snapshot-best")
    detector_path = _pick_best_snapshot(source_train_dir, "snapshot-detector-best")
    print(f"Using snapshot_path: {snapshot_path}")
    print(f"Using detector_path: {detector_path}")
    return str(snapshot_path), str(detector_path)


def delete_created_training_artifacts(project_path, shuffle, iteration=0):
    """Delete generated training-dataset and model artifacts for one shuffle.

    This removes:
    - Any file/folder containing "shuffle{shuffle}" under
      training-datasets/iteration-{iteration}/UnaugmentedDataSet_*
    - Corresponding shuffle entries in each metadata.yaml under
      training-datasets/iteration-{iteration}/UnaugmentedDataSet_*
    - Any model folder matching "*shuffle{shuffle}" under
      dlc-models-pytorch/iteration-{iteration}
    """
    project_path = Path(project_path)
    shuffle_token = f"shuffle{shuffle}"

    removed = {"dataset_entries": 0, "metadata_entries": 0, "model_dirs": 0}

    dataset_iteration_dir = project_path / "training-datasets" / f"iteration-{iteration}"
    if dataset_iteration_dir.exists():
        for unaug_dir in dataset_iteration_dir.glob("UnaugmentedDataSet_*"):
            for item in unaug_dir.iterdir():
                if shuffle_token not in item.name:
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                removed["dataset_entries"] += 1

            metadata_path = unaug_dir / "metadata.yaml"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = yaml.safe_load(f) or {}

                shuffles = metadata.get("shuffles", {})
                if isinstance(shuffles, dict):
                    to_delete = [
                        key
                        for key, value in shuffles.items()
                        if shuffle_token in str(key)
                        or (isinstance(value, dict) and value.get("index") == shuffle)
                    ]
                    for key in to_delete:
                        shuffles.pop(key, None)
                    if to_delete:
                        metadata["shuffles"] = shuffles
                        with open(metadata_path, "w", encoding="utf-8") as f:
                            yaml.safe_dump(metadata, f, sort_keys=False)
                        removed["metadata_entries"] += len(to_delete)

    model_iteration_dir = project_path / "dlc-models-pytorch" / f"iteration-{iteration}"
    if model_iteration_dir.exists():
        for model_dir in model_iteration_dir.glob(f"*{shuffle_token}"):
            if model_dir.is_dir():
                shutil.rmtree(model_dir)
                removed["model_dirs"] += 1

    print(
        f"Removed {removed['dataset_entries']} dataset entries, "
        f"{removed['metadata_entries']} metadata entries, and "
        f"{removed['model_dirs']} model directories for {shuffle_token}."
    )
    return removed
