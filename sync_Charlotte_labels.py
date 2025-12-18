import os
import shutil

new_label_path = "../rat_label"
project_path = "projects/rat_pose"
project_label_path = os.path.join(project_path, "labeled-data")

# copy and replace all label files from new_label_path to project_label_path
for session_folder in os.listdir(new_label_path):
    # skip files starting with '.'
    if session_folder.startswith('.'):
        continue
    if os.path.isdir(os.path.join(new_label_path, session_folder)):
        src_folder = os.path.join(new_label_path, session_folder)
        dst_folder = os.path.join(project_label_path, session_folder)
        if os.path.exists(dst_folder):
            shutil.rmtree(dst_folder)
        shutil.copytree(src_folder, dst_folder)
