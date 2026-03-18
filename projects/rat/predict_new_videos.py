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


from modules.kalman import kalman_video
from modules.kalman_fitter import kalman_fitter22  # 新增/替换

kalman_params, metrics = kalman_fitter22(
    project_path,
    shuffle=shuffle,
    trainingsetindex=0,
    use_cached_predictions=True,
    dispersion_visible_recall_target=0.99,
    test_fraction=0.2,
    test_seed=42,
    torch_device="cuda",  # 有GPU就用cuda；没有可改成"cpu"或去掉
)

kalman_video(
    config_path,
    videofile_path,
    shuffle=shuffle,
    destfolder=destfolder,
    # render_conf_min = 0.6,
    # max_extrapolation_frames = 10,
    render_conf_min_offset=-0.1,
    kalman_params=kalman_params,
    stability_n=61,
    stability_k=30,
)


from modules.inference_utils import generate_missing_label_heatmap

generate_missing_label_heatmap(
    project_path=project_path,
    shuffle=3,
    outpath=project_path+"/test_heat",   
    output_resolution=2048
)

out_file = generate_keypoint_heatmap_grid(
    project_path=project_path,                         
    shuffle=3,                                               
    video_path=videofile_path[0],   
    frame_idx=range(0, 1000, 100),                                            
    outpath=project_path+"/test_heat",                        
    # output_resolution=2048*8
    background_alpha = 0.6
)
print(out_file)