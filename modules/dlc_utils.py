
import re
import os
import math
import pandas as pd
import yaml
import deeplabcut
from modules.image_utils import get_image_from_video, save_image
from pathlib import Path
import copy


def load_config(project_path):
    config_path = os.path.join(project_path, 'config.yaml')
    fix_dlc_config(config_path)
    config = deeplabcut.auxiliaryfunctions.read_config(config_path)
    return config

def save_config(project_path, config):
    config_path = os.path.join(project_path, 'config.yaml')
    deeplabcut.auxiliaryfunctions.write_config(config_path, config)
    fix_dlc_config(config_path)


def fix_dlc_config(config_path):
    """Fix wrapped video_sets keys by joining them onto a single line.

    When paths contain spaces, YAML line-wrapping can split a single key
    across multiple lines, which breaks parsing. This function merges
    wrapped key lines into a single long line without introducing ?/:.
    It also normalizes legacy ?/: entries into plain "path: {crop...}".
    """
    with open(config_path, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    in_video_sets = False
    i = 0

    def _is_value_line(s):
        s = s.strip()
        if not s:
            return False
        if s.startswith('#'):
            return False
        if s.startswith('crop:'):
            return True
        if s.startswith(':'):
            return s.lstrip(':').lstrip().startswith('crop:')
        return False

    def _clean_key_part(s):
        s = s.strip()
        if s.startswith('?'):
            s = s[1:].lstrip()
        if s.startswith(':'):
            s = s[1:].lstrip()
        if s.endswith(':'):
            s = s[:-1]
        return s

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == 'video_sets:':
            in_video_sets = True
            fixed_lines.append(line)
            i += 1
            continue

        if in_video_sets:
            # Exit video_sets on non-indented, non-empty, non-comment line
            if stripped and not line[0].isspace():
                in_video_sets = False
                continue

            # Parse and normalize video_sets entries
            if line.startswith('  '):
                indent_str = '  '
                key_parts = []
                value_lines = []

                # Collect key parts until we hit a value line
                while i < len(lines):
                    curr = lines[i]
                    curr_stripped = curr.strip()
                    if curr_stripped and not curr.startswith(' '):
                        break
                    if not curr_stripped or curr_stripped.startswith('#'):
                        i += 1
                        continue
                    if _is_value_line(curr):
                        val = curr_stripped
                        if val.startswith(':'):
                            val = val[1:].lstrip()
                        if val.endswith(':'):
                            val = val[:-1].rstrip()
                        value_lines.append(f"    {val}\n")
                        i += 1
                        # Collect any additional value lines
                        while i < len(lines):
                            nxt = lines[i]
                            nxt_stripped = nxt.strip()
                            if not nxt.startswith('    ') or _is_value_line(nxt) is False:
                                break
                            val2 = nxt_stripped
                            if val2.startswith(':'):
                                val2 = val2[1:].lstrip()
                            if val2.endswith(':'):
                                val2 = val2[:-1].rstrip()
                            value_lines.append(f"    {val2}\n")
                            i += 1
                        break
                    key_parts.append(_clean_key_part(curr))
                    i += 1

                if key_parts:
                    key = ' '.join(key_parts)
                    fixed_lines.append(f"{indent_str}{key}:\n")
                    if value_lines:
                        fixed_lines.extend(value_lines)
                    continue

        fixed_lines.append(line)
        i += 1

    with open(config_path, 'w') as f:
        f.writelines(fixed_lines)




def remove_all_cache(project_path, type = ['.png', '.h5']):
    """
    Removes all image files and h5 files from the labeled-data directory.
    """
    labeled_data_path = os.path.join(project_path, 'labeled-data')
    if not os.path.exists(labeled_data_path):
        print(f"Labeled data directory not found: {labeled_data_path}")
        return

    for video_folder in os.listdir(labeled_data_path):
        video_folder_path = os.path.join(labeled_data_path, video_folder)
        if os.path.isdir(video_folder_path):
            cache_files = [f for f in os.listdir(video_folder_path) if f.endswith(tuple(type))]
            for cache_file in cache_files:
                # exclude machine labels like machinelabels-iter0.h5
                if not cache_file.startswith("machinelabels-iter"):
                    os.remove(os.path.join(video_folder_path, cache_file))


def reconstruct_labeled_data(project_path, refresh=False):
    """
    Reconstructs labeled data by extracting frames from videos based on existing CSV files.
    
    Args:
        project_path (str): Path to the DeepLabCut project directory
    """
    labeled_data_path = os.path.join(project_path, 'labeled-data')
    videos_path = os.path.join(project_path, 'videos')
    
    # Check if labeled-data directory exists
    if not os.path.exists(labeled_data_path):
        print(f"Labeled data directory not found: {labeled_data_path}")
        return
    
    # Get all video folders in labeled-data directory
    labelled_folders = [f for f in os.listdir(labeled_data_path) 
                    if os.path.isdir(os.path.join(labeled_data_path, f))]
    
    video_files_all = [f for f in os.listdir(videos_path) if f.endswith(('.mp4', '.mkv'))]
    video_names = [os.path.splitext(f)[0] for f in video_files_all]
    
    # Match video folders with video files
    video_folders = []
    video_files = []
    for i in range(len(video_names)):
        if video_names[i] in labelled_folders:
            video_folders.append(video_names[i])
            video_files.append(video_files_all[i])
    
    print(f"Found {len(video_folders)} video folders in labeled-data directory")
    
    for i in range(len(video_folders)):
        video_folder = video_folders[i]
        video_file = video_files[i]
        print(f"\nProcessing video folder: {video_folder}")
        label_folder_path = os.path.join(labeled_data_path, video_folder)
        video_path = os.path.join(videos_path, video_file)
        
        png_files = [f for f in os.listdir(label_folder_path) if f.endswith('.png')]
        if refresh:
            # delete the existing png files in the label folder
            for png_file in png_files:
                os.remove(os.path.join(label_folder_path, png_file))
        
        
        # Find CSV file in the video folder
        csv_files = [f for f in os.listdir(label_folder_path) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"  No CSV file found in {video_folder}")
            continue
        
        for j in range(len(csv_files)):
            print(f"  Processing CSV file: {csv_files[j]}")
            csv_path = os.path.join(label_folder_path, csv_files[j])
            numer_frames = csv_to_img(label_folder_path, video_path, csv_path, refresh=refresh)
            print(f"    Extracted {numer_frames} frames")


def csv_to_img(label_folder_path, video_path, csv_path, refresh=False):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        image_frames_char = df['Unnamed: 2'].tolist()
        # remove na values
        image_frames_char = [i for i in image_frames_char if isinstance(i, str)]
        # remove duplicates
        if not refresh:
            png_files = [f for f in os.listdir(label_folder_path) if f.endswith('.png')]
            image_frames_char = list(set(image_frames_char) - set(png_files))
        
        # Extract frame indices from image filenames
        for frame_filename in image_frames_char:
            # Extract frame index from filename (format: img{frame_idx}.png)
            match = re.match(r'img(\d+)\.png', frame_filename)
            if match:
                frame_idx = int(match.group(1))
                frame = get_image_from_video(video_path, frame_idx)
                output_path = os.path.join(label_folder_path, frame_filename)
                save_image(output_path, frame)
            else:
                print(f"  Warning: Could not extract frame index from: {frame_filename}")
        return len(image_frames_char)
    except Exception as e:
        print(f"  Error processing {csv_path}: {e}")
    return 0



def pack_h5_data(project_path):
    """
    Prepare the h5 data for training
    """
    remove_all_cache(project_path, type=['.h5'])
    config_path = os.path.join(project_path, "config.yaml")
    config = load_config(project_path)
    deeplabcut.convertcsv2h5(config_path, userfeedback=False)
    
    

def change_video_name(project_path, old_name, new_name):
    config = load_config(project_path)
    video_sets = config['video_sets']
    old_video_path = os.path.join(os.getcwd(), project_path, 'videos', f"{old_name}.mp4")
    video_key = list(video_sets.keys())
    new_video_path = None
    for key in video_key:
        if key.endswith(f"\\{old_name}.mp4"):
            # update config file
            new_video_path = os.path.join(os.getcwd(), project_path, 'videos', f"{new_name}.mp4")
            new_video_path = new_video_path.replace("\\", "/")
            video_sets[new_video_path] = video_sets.pop(key)
            config['video_sets'] = video_sets
            save_config(project_path, config)
    
    if new_video_path is None:
        print(f"Video name {old_name} not found in config.yaml")
        
    # rename the video file
    if os.path.exists(old_video_path):
        new_video_path = os.path.join(os.getcwd(), project_path, 'videos', f"{new_name}.mp4")
        os.rename(old_video_path, new_video_path)
    else:
        print(f"Video file {old_video_path} not found")
    
    # change the labels file name
    labels_dir = os.path.join(project_path, 'labeled-data')
    old_dir = os.path.join(labels_dir, old_name)
    new_dir = os.path.join(labels_dir, new_name)
    if os.path.exists(old_dir):
        os.rename(old_dir, new_dir)
    else:
        print(f"Labels directory {old_dir} not found")
    
    # read the csv file and change the video name in column B
    if os.path.exists(new_dir):
        csv_files = [f for f in os.listdir(new_dir) if f.endswith('.csv')]
        for csv_file in csv_files:
            csv_path = os.path.join(new_dir, csv_file)
            df = pd.read_csv(csv_path, header=None)
            df.iloc[3:, 1] = new_name
            df.to_csv(csv_path, index=False, header=False)
    else:
        print(f"Labels directory {new_dir} not found")



def get_full_path(p, base=None):
    """
    Return an absolute path for p. If p is relative and base is provided,
    base is used as the anchor. Uses resolve(strict=False) so it works even
    if the target doesn't exist yet.
    """
    path = Path(p)
    if base and not path.is_absolute():
        path = Path(base) / path
    return str(path.resolve(strict=False))

def rebase_project(project_path):
    full_project_path = get_full_path(project_path)
    config = load_config(project_path)
    if 'project_path' in config:
        config['project_path'] = full_project_path
        
    video_sets = config['video_sets']
    video_keys = list(video_sets.keys())
    # regular expression to capture videos/XXX.mp4 or XXX.mkv
    pattern = r"videos[\\/](?P<name>[^\\/]+\.(?:mp4|mkv))"
    video_names = []
    for key in video_keys:
        video_names.extend(re.findall(pattern, key, flags=re.IGNORECASE))
    
    new_video_set = {}
    for i in range(len(video_keys)):
        new_video_key = os.path.join(full_project_path, "videos", video_names[i])
        new_video_set[new_video_key] = video_sets[video_keys[i]]
    
    config["video_sets"] = new_video_set
    save_config(project_path, config)
    
    
def stat_report(project_path):
    """Report labeling statistics for each video in the project.
    
    Only reports on videos listed in config.yaml video_sets.
    For each video folder in labeled-data/, reports:
      - Manual labels: {#images with matching PNGs}/{#total label entries in CSV}
      - Machine labels: {#images with matching PNGs}/{#total label entries in CSV}
      - Uncorrected: machine-labeled images that do NOT appear in manual labels
    
    Last line prints the total across all videos.
    """
    config = load_config(project_path)
    labeled_data_path = os.path.join(project_path, 'labeled-data')
    
    if not os.path.exists(labeled_data_path):
        print(f"Labeled data directory not found: {labeled_data_path}")
        return
    
    # Extract video folder names from config video_sets keys
    video_sets = config.get('video_sets', {})
    folders = sorted([
        Path(video_path).stem for video_path in video_sets.keys()
    ])
    
    total_videos = 0
    total_manual_matched = 0
    total_manual_labels = 0
    total_machine_matched = 0
    total_machine_labels = 0
    total_uncorrected = 0
    
    print(f"{'Video':<35} {'Manual (img/label)':<20} {'Machine (img/label)':<20} {'Uncorrected':<12}")
    print("-" * 87)
    
    for folder in folders:
        folder_path = os.path.join(labeled_data_path, folder)
        if not os.path.isdir(folder_path):
            print(f"{folder:<35} {'(folder missing)':<20}")
            continue
        
        # Collect PNG filenames
        png_files = {f for f in os.listdir(folder_path) if f.endswith('.png')}
        
        # Parse manual labels from CollectedData CSV
        manual_csv = os.path.join(folder_path, 'CollectedData_rats.csv')
        manual_images = set()
        n_manual_labels = 0
        if os.path.exists(manual_csv):
            df = pd.read_csv(manual_csv, header=None, skiprows=3)
            manual_images = set(df.iloc[:, 2].dropna().tolist())
            n_manual_labels = len(manual_images)
        
        # Parse machine labels from machinelabels CSV
        machine_csv = os.path.join(folder_path, 'machinelabels.csv')
        machine_images = set()
        n_machine_labels = 0
        if os.path.exists(machine_csv):
            df = pd.read_csv(machine_csv, header=None, skiprows=3)
            machine_images = set(df.iloc[:, 2].dropna().tolist())
            n_machine_labels = len(machine_images)
        
        # Matched = label images that have a corresponding PNG
        n_manual_matched = len(manual_images & png_files)
        n_machine_matched = len(machine_images & png_files)
        
        # Uncorrected = machine-labeled images that do NOT have manual labels
        n_uncorrected = len(machine_images - manual_images)
        
        total_videos += 1
        total_manual_matched += n_manual_matched
        total_manual_labels += n_manual_labels
        total_machine_matched += n_machine_matched
        total_machine_labels += n_machine_labels
        total_uncorrected += n_uncorrected
        
        manual_str = f"{n_manual_matched}/{n_manual_labels}"
        machine_str = f"{n_machine_matched}/{n_machine_labels}"
        print(f"{folder:<35} {manual_str:<20} {machine_str:<20} {n_uncorrected:<12}")
    
    print("-" * 87)
    manual_total = f"{total_manual_matched}/{total_manual_labels}"
    machine_total = f"{total_machine_matched}/{total_machine_labels}"
    print(f"{'TOTAL (' + str(total_videos) + ' videos)':<35} {manual_total:<20} {machine_total:<20} {total_uncorrected:<12}")


def _load_xy_from_label_csv(csv_path):
    """
    Load a DLC-style CSV and return a frame-indexed DataFrame of bodypart x/y columns.
    If duplicate bodypart/coord columns exist across scorers, the first non-null value is used.
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    if df.shape[1] < 4:
        return pd.DataFrame()

    frame_series = df.iloc[:, 2].astype(str).str.strip()
    frame_series = frame_series.where(frame_series != "nan")

    col_map = {}
    for col in df.columns[3:]:
        # DeepLabCut header columns are usually (scorer, bodypart, coord)
        if len(col) < 3:
            continue
        bodypart = str(col[1]).strip()
        coord = str(col[2]).strip().lower()
        if coord not in ("x", "y"):
            continue
        col_map.setdefault((bodypart, coord), []).append(col)

    out = pd.DataFrame({"frame": frame_series})
    for (bodypart, coord), cols in col_map.items():
        values = df.loc[:, cols]
        if isinstance(values, pd.Series):
            out[f"{bodypart}_{coord}"] = values
        else:
            out[f"{bodypart}_{coord}"] = values.bfill(axis=1).iloc[:, 0]

    out = out.dropna(subset=["frame"]).set_index("frame")
    # Keep one row per frame if duplicates exist.
    out = out.groupby(level=0).first()
    return out


def find_unchanged_labels(project_path, cutoff=1):
    """
    Compare manual labels (CollectedData_rats.csv) vs machine labels (machinelabels.csv)
    across labeled-data video folders.

    For frames present in both files, report only if all shared
    keypoints are within `cutoff` Euclidean distance.
    Missing/non-missing mismatch for any keypoint coordinate is treated as larger than cutoff.

    Returns:
        list[dict]: each item includes:
            - video_name
            - frame_name
            - manual_idx (0-based index in CollectedData_rats.csv data rows)
            - manual_csv
    """
    labeled_data_path = os.path.join(project_path, "labeled-data")
    if not os.path.isdir(labeled_data_path):
        print(f"Labeled data directory not found: {labeled_data_path}")
        return []

    unchanged_frames = []

    for video_name in sorted(os.listdir(labeled_data_path)):
        video_dir = os.path.join(labeled_data_path, video_name)
        if not os.path.isdir(video_dir):
            continue

        manual_csv = os.path.join(video_dir, "CollectedData_rats.csv")
        machine_csv = os.path.join(video_dir, "machinelabels.csv")
        if not (os.path.exists(manual_csv) and os.path.exists(machine_csv)):
            continue

        manual_df = _load_xy_from_label_csv(manual_csv)
        machine_df = _load_xy_from_label_csv(machine_csv)
        if manual_df.empty or machine_df.empty:
            continue
        
        # Manual row index mapping in original CollectedData order (0..N-1).
        manual_raw = pd.read_csv(manual_csv, header=[0, 1, 2])
        manual_frames_raw = manual_raw.iloc[:, 2].astype(str).str.strip()
        manual_frames_raw = manual_frames_raw.where(manual_frames_raw != "nan")
        manual_idx_map = {}
        for idx, frame in enumerate(manual_frames_raw.dropna()):
            if frame not in manual_idx_map:
                manual_idx_map[frame] = idx

        manual_bps = {
            c[:-2] for c in manual_df.columns
            if c.endswith("_x") and f"{c[:-2]}_y" in manual_df.columns
        }
        machine_bps = {
            c[:-2] for c in machine_df.columns
            if c.endswith("_x") and f"{c[:-2]}_y" in machine_df.columns
        }
        shared_bps = manual_bps & machine_bps
        if not shared_bps:
            continue

        shared_frames = manual_df.index.intersection(machine_df.index)
        for frame_name in shared_frames:
            manual_row = manual_df.loc[frame_name]
            machine_row = machine_df.loc[frame_name]
            frame_is_unchanged = True

            for bp in shared_bps:
                mx = manual_row.get(f"{bp}_x")
                my = manual_row.get(f"{bp}_y")
                px = machine_row.get(f"{bp}_x")
                py = machine_row.get(f"{bp}_y")

                m_missing = pd.isna(mx) or pd.isna(my)
                p_missing = pd.isna(px) or pd.isna(py)

                # Missing/non-missing mismatch counts as a large difference.
                if m_missing != p_missing:
                    frame_is_unchanged = False
                    break

                # If both are missing for this keypoint, treat as unchanged for this keypoint.
                if m_missing and p_missing:
                    continue

                dist = math.hypot(float(mx) - float(px), float(my) - float(py))
                if dist >= cutoff:
                    frame_is_unchanged = False
                    break

            if frame_is_unchanged:
                manual_idx = manual_idx_map.get(frame_name)
                unchanged_frames.append({
                    "video_name": video_name,
                    "frame_name": frame_name,
                    "manual_idx": manual_idx,
                    "manual_csv": manual_csv,
                })
                print(f"{video_name}, {frame_name}, manual_idx={manual_idx}")

    print(f"Total unchanged frames: {len(unchanged_frames)}")
    return unchanged_frames


def remove_unchanged_labels(unchanged):
    """
    Remove unchanged label rows from manual CSV files using find_unchanged_labels output.

    This function removes raw CSV lines by line number (header_rows + manual_idx),
    which preserves original file structure and empty-cell formatting.
    """
    if not unchanged:
        print("No unchanged labels provided.")
        return

    header_rows = 3
    remove_map = {}

    for item in unchanged:
        if isinstance(item, dict):
            manual_csv = item.get("manual_csv")
            manual_idx = item.get("manual_idx")
        elif isinstance(item, (tuple, list)) and len(item) >= 4:
            # Backward-compatible tuple/list support: (video_name, frame_name, manual_idx, manual_csv)
            manual_idx = item[2]
            manual_csv = item[3]
        else:
            continue

        if manual_csv is None or manual_idx is None:
            continue

        if not isinstance(manual_idx, int):
            try:
                manual_idx = int(manual_idx)
            except (TypeError, ValueError):
                continue

        remove_map.setdefault(manual_csv, set()).add(manual_idx)

    total_removed = 0
    for manual_csv, idx_set in remove_map.items():
        if not os.path.exists(manual_csv):
            print(f"File not found, skip: {manual_csv}")
            continue

        with open(manual_csv, "r", newline="") as f:
            lines = f.readlines()

        line_nums_to_remove = {
            header_rows + idx
            for idx in idx_set
            if idx >= 0 and (header_rows + idx) < len(lines)
        }

        if not line_nums_to_remove:
            print(f"No valid rows to remove: {manual_csv}")
            continue

        new_lines = [
            line for line_no, line in enumerate(lines)
            if line_no not in line_nums_to_remove
        ]

        with open(manual_csv, "w", newline="") as f:
            f.writelines(new_lines)

        removed_count = len(line_nums_to_remove)
        total_removed += removed_count
        print(f"Removed {removed_count} rows from {manual_csv}")

    print(f"Total removed rows: {total_removed}")


# set the augmentation probability in the model config
def set_transform_prob(model_cfg, prob=0.2):
    model_cfg = copy.deepcopy(model_cfg)
    trans = model_cfg["data"]["train"]['transform']
    for i in range(len(trans)):
        item = trans[i]
        if 'p' in item:
            item['p'] = prob
    return model_cfg
    
