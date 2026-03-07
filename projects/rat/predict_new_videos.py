import deeplabcut
import os
from deeplabcut.modelzoo import build_weight_init
import shutil
superanimal_name = 'superanimal_quadruped'
project_path = 'projects/rat'
config_path = os.path.join(project_path, "config.yaml")


# Enter the list of videos to analyze.
shuffle=3
videofile_path = [
    os.path.join(project_path, "test_snap/Camera4_stitched_600_660.mp4")]
videofile_path = [os.path.abspath(i) for i in videofile_path]
deeplabcut.analyze_videos(config_path, videofile_path, shuffle=shuffle)
# deeplabcut.create_labeled_video(config_path, videofile_path, draw_skeleton=True, pcutoff=0.8, overwrite=True, shuffle=shuffle)


deeplabcut.filterpredictions(
    config_path,
    videofile_path,
    shuffle=shuffle,
    filtertype="median",
    windowlength = 7
)

deeplabcut.create_labeled_video(
    config_path,
    videofile_path,
    shuffle=shuffle,
    draw_skeleton=True,
    pcutoff=0.6,
    filtered=True,   # key flag: use filtered poses
    overwrite=True,
)