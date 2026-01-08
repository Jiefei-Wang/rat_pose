import os
from modules.dlc_utils import remove_all_cache, reconstruct_labeled_data, pack_h5_data, change_video_name, rebase_project


project_path = "projects/rat"

###############################
# Rebase project
# make sure the video path is correct
###############################
rebase_project(project_path)


###############################
# 1. rebuild the images
# 2. create h5
###############################
remove_all_cache(project_path)
reconstruct_labeled_data(project_path)
pack_h5_data(project_path)



###############################
# verify labels
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