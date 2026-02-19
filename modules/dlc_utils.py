
import re
import os
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
    """Fix video_sets entries missing explicit YAML key (?) notation.
    
    When paths contain spaces and wrap across lines, ruamel.yaml requires
    the explicit ?/: key-value notation. This fixes entries that omit it.
    """
    with open(config_path, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    in_video_sets = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == 'video_sets:':
            in_video_sets = True
            fixed_lines.append(line)
            i += 1
            continue

        # Exit video_sets on non-indented, non-empty, non-comment line
        if in_video_sets and stripped and not line[0].isspace():
            in_video_sets = False

        if in_video_sets and stripped.startswith('/'):
            # Found a path key without explicit ? notation
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent

            if stripped.endswith(':'):
                # Single-line path key: "  /path/file.mp4:"
                fixed_lines.append(f"{indent_str}? {stripped[:-1]}\n")
                i += 1
            else:
                # Multi-line path key: "  /path/start...\n    ...end.mp4:"
                fixed_lines.append(f"{indent_str}? {line.lstrip()}")
                i += 1
                while i < len(lines):
                    cont = lines[i]
                    if cont.strip().endswith(':'):
                        trimmed = cont.rstrip().rstrip('\n')
                        fixed_lines.append(f"{trimmed[:-1]}\n")
                        i += 1
                        break
                    else:
                        fixed_lines.append(cont)
                        i += 1

            # Next line is the value (e.g., "    crop: ...")
            if i < len(lines):
                val = lines[i].strip()
                fixed_lines.append(f"{indent_str}: {val}\n")
                i += 1
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
      - Corrected: machine-labeled images that also appear in manual labels
    
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
    total_corrected = 0
    
    print(f"{'Video':<35} {'Manual (img/label)':<20} {'Machine (img/label)':<20} {'Corrected':<10}")
    print("-" * 85)
    
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
        
        # Corrected = machine-labeled images that also have manual labels
        n_corrected = len(machine_images & manual_images)
        
        total_videos += 1
        total_manual_matched += n_manual_matched
        total_manual_labels += n_manual_labels
        total_machine_matched += n_machine_matched
        total_machine_labels += n_machine_labels
        total_corrected += n_corrected
        
        manual_str = f"{n_manual_matched}/{n_manual_labels}"
        machine_str = f"{n_machine_matched}/{n_machine_labels}"
        print(f"{folder:<35} {manual_str:<20} {machine_str:<20} {n_corrected:<10}")
    
    print("-" * 85)
    manual_total = f"{total_manual_matched}/{total_manual_labels}"
    machine_total = f"{total_machine_matched}/{total_machine_labels}"
    print(f"{'TOTAL (' + str(total_videos) + ' videos)':<35} {manual_total:<20} {machine_total:<20} {total_corrected:<10}")


# set the augmentation probability in the model config
def set_transform_prob(model_cfg, prob=0.2):
    model_cfg = copy.deepcopy(model_cfg)
    trans = model_cfg["data"]["train"]['transform']
    for i in range(len(trans)):
        item = trans[i]
        if 'p' in item:
            item['p'] = prob
    return model_cfg
    
