import os
import shutil
from pathlib import Path
import glob

project_path = 'projects/rat_pose'
destination_path = f'G:/My Drive/code/rat_pose/{project_path}'
# move everything in this folder, except for video files, to google drive
if not os.path.exists(destination_path):
    os.makedirs(destination_path)
    
for item in os.listdir(project_path):
    s = os.path.join(project_path, item)
    d = os.path.join(destination_path, item)
    if os.path.isdir(s):
        if not os.path.exists(d):
            shutil.copytree(s, d, ignore=shutil.ignore_patterns('*.mp4', '*.avi', '*.mov', '*.mkv'))
    else:
        if not s.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            shutil.copy2(s, d)
            