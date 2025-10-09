import json
import cv2
import deeplabcut
import yaml
import os

deeplab_working_directory = "config"
project_name = 'Sleap_Rat_test'
video_folder = r"C:\Users\jiefei\OneDrive1\OneDrive\UTMB\teaching\summer institute\student projects\2025\Tom\Green_Project_files\video"

# Loading in skeleton json file that was previously created to add to config file
with open('output\\sleap_skeleton\\sleap_data.json', 'r') as f:
    skeleton_data = json.load(f)

sleap_video_filenames = skeleton_data['videos']
sleap_video_filenames = [os.path.basename(video) for video in sleap_video_filenames]
sleap_video_filenames_noext = [os.path.splitext(video)[0] for video in sleap_video_filenames]


videos = [os.path.join(video_folder, sleap_video_filenames[i]) for i in range(len(sleap_video_filenames))]

# New project creation (only need to do once)
config_path = deeplabcut.create_new_project(
        project_name, # Name of Project
        experimenter  = 'T',  # Name of scorer
        videos = videos,
        working_directory= deeplab_working_directory,
        copy_videos = True
)    
