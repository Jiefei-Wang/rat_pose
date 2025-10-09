from modules.dlc_utils import remove_all_cache, reconstruct_labeled_data, load_config, save_config, pack_h5_data

project_path = "projects/rat_pose"

###############################
# 1. delete all images and h5 data
# 2. rebuild the images
# 3. create h5
###############################
remove_all_cache(project_path)
reconstruct_labeled_data(project_path)
pack_h5_data(project_path)