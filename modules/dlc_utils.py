
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
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(project_path, config):
    config_path = os.path.join(project_path, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

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
        
        for i in range(len(csv_files)):
            print(f"  Processing CSV file: {csv_files[i]}")
            csv_path = os.path.join(label_folder_path, csv_files[i])
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
    
    
# set the augmentation probability in the model config
def set_transform_prob(model_cfg, prob=0.2):
    model_cfg = copy.deepcopy(model_cfg)
    trans = model_cfg["data"]["train"]['transform']
    for i in range(len(trans)):
        item = trans[i]
        if 'p' in item:
            item['p'] = prob
    return model_cfg
    
