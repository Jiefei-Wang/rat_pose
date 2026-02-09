import deeplabcut
import os
from deeplabcut.modelzoo import build_weight_init
import shutil
from modules.dlc_utils import load_config
shuffle=2
project_path = 'projects/rat'
config_path = os.path.abspath(os.path.join(project_path, "config.yaml"))

config = load_config(project_path)
video_sets = config['video_sets']
video_path = list(video_sets.keys())


output_folder = os.path.join(project_path, f"labeled_video", str(shuffle))

# to full path
videofile_paths = [os.path.abspath(i) for i in video_path]
output_folder = os.path.abspath(output_folder)

# analyze videos
deeplabcut.analyze_videos(config_path, videofile_paths, shuffle=shuffle, destfolder  = output_folder)


# create tracked video
deeplabcut.create_labeled_video(config_path, videofile_paths, draw_skeleton=True, pcutoff=0.9, overwrite=True, shuffle=shuffle, destfolder  = output_folder)

# extract outlier frames
deeplabcut.extract_outlier_frames(config_path, videofile_paths, outlieralgorithm="jump", shuffle=shuffle, destfolder= output_folder, automatic =True)