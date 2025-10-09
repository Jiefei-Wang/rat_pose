import deeplabcut
import os
superanimal_name = 'superanimal_quadruped'
deeplabcut_path = 'projects/rat_pose'
config_path = os.path.join(deeplabcut_path, "config.yaml")
deeplabcut.label_frames()