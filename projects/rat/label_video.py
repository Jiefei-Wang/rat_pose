import deeplabcut
import os
from deeplabcut.modelzoo import build_weight_init
import shutil
superanimal_name = 'superanimal_quadruped'
project_path = 'projects/rat'
config_path = os.path.abspath(os.path.join(project_path, "config.yaml"))


test_folder = os.path.abspath(os.path.join(project_path, "videos/"))

# Find all video files in the folder
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
video_files = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f)) and os.path.splitext(f)[1].lower() in video_extensions]

videofile_paths = [os.path.abspath(os.path.join(test_folder, video_file)) for video_file in video_files]

shuffle=1
output_folder = os.path.join(project_path, "labeled_video", str(shuffle))

# to full path
output_folder = os.path.abspath(output_folder)


# analyze videos
deeplabcut.analyze_videos(config_path, videofile_paths, shuffle=shuffle, destfolder  = output_folder)


# create tracked video
deeplabcut.create_labeled_video(config_path, videofile_paths, draw_skeleton=True, pcutoff=0.9, overwrite=True, shuffle=shuffle, destfolder  = output_folder)

# extract outlier frames
deeplabcut.extract_outlier_frames(config_path, videofile_paths, outlieralgorithm="jump", shuffle=shuffle, destfolder= output_folder, automatic =True)