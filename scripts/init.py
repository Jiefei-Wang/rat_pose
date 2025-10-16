import os
from modules.dlc_utils import remove_all_cache, reconstruct_labeled_data, pack_h5_data, change_video_name, rebase_project



project_path = "projects/rat_pose"

###############################
# Rebase project
# make sure the video path is correct
###############################
rebase_project(project_path)


###############################
# Simplify video name.
###############################
change_video_name(project_path, "Video_Generation_Rat_in_Chamber", "ai1")
change_video_name(project_path, "Rat_Tracking_Video_Generation_Request (3)", "ai2")
change_video_name(project_path, "Rat_Tracking_Video_Generation_Request (2)", "ai3")
change_video_name(project_path, "Rat_Video_Generation_Request_Fulfilled", "ai4")
change_video_name(project_path, "Rat_Tracking_Video_Generation_Request (1)", "ai5")
change_video_name(project_path, "Rat_Video_Generated_Successfully", "ai6")
change_video_name(project_path, "Video_for_Rat_Tracking_Training (3)", "ai7")
change_video_name(project_path, "Rat_Tracking_Video_Generation_Request", "ai8")


    



###############################
# 1. delete all images and h5 data
# 2. rebuild the images
# 3. create h5
###############################
remove_all_cache(project_path)
reconstruct_labeled_data(project_path)
pack_h5_data(project_path)


