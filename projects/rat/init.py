import os
from modules.dlc_utils import remove_all_cache, reconstruct_labeled_data, pack_h5_data, change_video_name, rebase_project,stat_report


project_path = "projects/rat"

###############################
# Rebase project so the video path is correct
# This always assume your videos are in '{project_path}/videos'
###############################
rebase_project(project_path)

stat_report(project_path)

###############################
# 1. create h5
# 2. extract images from videos based on labels
###############################
remove_all_cache(project_path)
reconstruct_labeled_data(project_path)
pack_h5_data(project_path)



###############################
# create images with labels
###############################
import shutil
# delete all existing label images
folders = [f for f in os.listdir(os.path.join(project_path, "labeled-data")) if f.endswith("_labeled")]
for folder in folders:
    folder_path = os.path.join(project_path, "labeled-data", folder)
    shutil.rmtree(folder_path)
    
import deeplabcut
config_path = os.path.join(project_path, "config.yaml")
deeplabcut.check_labels(config_path)