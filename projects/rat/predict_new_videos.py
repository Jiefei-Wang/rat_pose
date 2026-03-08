import deeplabcut
import os
from deeplabcut.modelzoo import build_weight_init
from modules.inference_utils import remove_filtered_prediction_files
from modules.kalman import kalman_video,kalman_fitter
import shutil
superanimal_name = 'superanimal_quadruped'
project_path = 'projects/rat'
config_path = os.path.join(project_path, "config.yaml")
destfolder = os.path.join(project_path, "test_snap")
destfolder = os.path.abspath(destfolder)

# Enter the list of videos to analyze.
shuffle=3
videofile_path = [
    os.path.join(project_path, "test_video/Camera4_stitched_600_660.mp4")]
videofile_path = [os.path.abspath(i) for i in videofile_path]
deeplabcut.analyze_videos(config_path, videofile_path, shuffle=shuffle, destfolder=destfolder)
# deeplabcut.create_labeled_video(config_path, videofile_path, draw_skeleton=True, pcutoff=0.8, overwrite=True, shuffle=shuffle)

remove_filtered_prediction_files(
    config=config_path,
    video=videofile_path,
    shuffle=shuffle,
    destfolder=destfolder
)
deeplabcut.filterpredictions(
    config=config_path,
    video=videofile_path,
    shuffle=shuffle,
    filtertype="median",
    windowlength = 7,
    destfolder=destfolder
)

deeplabcut.create_labeled_video(
    config=config_path,
    videos=videofile_path,
    shuffle=shuffle,
    draw_skeleton=True,
    pcutoff=0.8,
    filtered=False,   # key flag: use filtered poses
    overwrite=True,
    plot_bboxes = True,
    destfolder=destfolder
)


kalman_params = kalman_fitter(
    project_path,
    shuffle=shuffle,
    trainingsetindex=0,
    use_cached_predictions=True
)
kalman_video(
    config_path,
    videofile_path,
    shuffle=shuffle,
    destfolder=destfolder,
    # max_extrapolation_frames =3,
    # render_conf_min =0.3
    render_conf_min_offset = -0.3,
    gate_threshold_offset = 8,
    kalman_params = kalman_params
)
