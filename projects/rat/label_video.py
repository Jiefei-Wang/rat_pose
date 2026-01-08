import deeplabcut
import os
from deeplabcut.modelzoo import build_weight_init
import shutil
superanimal_name = 'superanimal_quadruped'
project_path = 'projects/rat2'
config_path = os.path.abspath(os.path.join(project_path, "config.yaml"))


test_folder = os.path.abspath(os.path.join(project_path, "test_video/"))
video_files = [
    "RAT 6 FR1 10-03-25 - Trim.mp4",
    "Camera4_stitched - Trim.mp4"
]
videofile_paths = [os.path.abspath(os.path.join(test_folder, video_file)) for video_file in video_files]

shuffle=1
output_folder = os.path.join(project_path, "labeled_video", str(shuffle))
# to full path
output_folder = os.path.abspath(output_folder)

deeplabcut.analyze_videos(config_path, videofile_paths, shuffle=shuffle, destfolder  = output_folder)
deeplabcut.create_labeled_video(config_path, videofile_paths, draw_skeleton=True, pcutoff=0.9, overwrite=True, shuffle=shuffle, destfolder  = output_folder)