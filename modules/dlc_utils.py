
import re
import os
import pandas as pd
import yaml
import deeplabcut
from modules.image_utils import get_image_from_video, save_image

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
                os.remove(os.path.join(video_folder_path, cache_file))


def reconstruct_labeled_data(project_path):
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
    
    video_files = [f for f in os.listdir(videos_path) if f.endswith('.mp4')]
    video_names = [os.path.splitext(f)[0] for f in video_files]
    
    # Make sure only process video folders that have corresponding video files
    video_folders = [f for f in labelled_folders if f in video_names]
    
    
    print(f"Found {len(video_folders)} video folders in labeled-data directory")
    
    for video_folder in video_folders:
        print(f"\nProcessing video folder: {video_folder}")
        label_folder_path = os.path.join(labeled_data_path, video_folder)
        video_path = os.path.join(videos_path, f"{video_folder}.mp4")
        
        # delete the existing png files in the label folder
        png_files = [f for f in os.listdir(label_folder_path) if f.endswith('.png')]
        for png_file in png_files:
            os.remove(os.path.join(label_folder_path, png_file))
        
        # Find CSV file in the video folder
        csv_files = [f for f in os.listdir(label_folder_path) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"  No CSV file found in {video_folder}")
            continue
        elif len(csv_files) > 1:
            print(f"  Multiple CSV files found in {video_folder}, using first one: {csv_files[0]}")
        
        csv_file = csv_files[0]
        csv_path = os.path.join(label_folder_path, csv_file)
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            image_frames = df['Unnamed: 2'].tolist()
            image_frames = [i for i in image_frames if isinstance(i, str)]
            
            # Extract frame indices from filenames (format: img{frame_idx}.png)
            frame_indices = []
            for frame in image_frames:
                match = re.match(r'img(\d+)\.png', frame)
                if match:
                    frame_idx = int(match.group(1))
                    frame_indices.append(frame_idx)
                else:
                    raise ValueError(f"Could not extract frame index from: {frame} in {video_folder}")

            for frame_idx in frame_indices:
                frame = get_image_from_video(video_path, frame_idx)
                output_path = os.path.join(label_folder_path, f"img{frame_idx:03d}.png")
                save_image(output_path, frame)
                
        except Exception as e:
            print(f"  Error processing {video_folder}: {e}")
            continue
    


def pack_h5_data(project_path):
    """
    Prepare the h5 data for training
    """
    config_path = os.path.join(project_path, "config.yaml")
    config = load_config(project_path)
    deeplabcut.convertcsv2h5(config_path, userfeedback=False, scorer=config['scorer'])
    
    

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

