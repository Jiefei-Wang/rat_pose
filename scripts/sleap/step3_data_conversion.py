import json
import cv2
import deeplabcut
import yaml
import os
import pandas as pd
import csv
from modules.image_utils import get_image_from_video, show_image, save_image
from modules.sleap_conversion import convert_sleap_to_dlc_format
from modules.dlc_utils import remove_all_cache, reconstruct_labeled_data, load_config, save_config

deeplab_working_directory = "config"
project_name = 'Sleap_Rat_test'

deeplabcut_path = 'projects/rat_pose'
config_path = os.path.join(deeplabcut_path, 'config.yaml')
sleap_path = "output/sleap_skeleton"




# Load config file to add in custom body parts and skeleton to the config file
config = load_config(deeplabcut_path)
# full path to all videos
all_videos = [os.path.join(deeplabcut_path, 'videos', f) for f in os.listdir(os.path.join(deeplabcut_path, 'videos')) if f.endswith('.mp4')]
# relative to full path
all_videos = [os.path.abspath(video) for video in all_videos]

config['video_sets'] = {i:{'crop': '0, 1280, 0, 720'} for i in all_videos}



# Skeleton and bodypart data are being set to the config files bodypart and skeleton
config['bodyparts'] = skeleton_data['bodyparts']
config['skeleton'] = skeleton_data['skeleton']


# Save updated config file
save_config(deeplabcut_path, config)



# Loading in skeleton json file that was previously created to add to config file
with open('output\\sleap_skeleton\\sleap_data.json', 'r') as f:
    skeleton_data = json.load(f)

sleap_video_filenames = skeleton_data['videos']
sleap_video_filenames = [os.path.basename(video) for video in sleap_video_filenames]
sleap_video_filenames_noext = [os.path.splitext(video)[0] for video in sleap_video_filenames]

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)


## SLEAP labels to csv format required by DeepLabCut
for i in range(len(sleap_video_filenames_noext)):
    video_name = sleap_video_filenames_noext[i]
    sleap_df = pd.read_csv(os.path.join(sleap_path, f"{video_name}.csv"))
    converted = convert_sleap_to_dlc_format(video_name, sleap_df, config)
    
    output_path = os.path.join(deeplabcut_path, 'labeled-data', video_name, f'CollectedData_{config["scorer"]}.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(converted)

