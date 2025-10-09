import deeplabcut
import os
from deeplabcut.modelzoo import build_weight_init
import shutil

superanimal_name = 'superanimal_quadruped'
project_path = 'projects/rat_pose'

config_path = os.path.join(project_path, "config.yaml")

weight_init = build_weight_init(
            cfg = config_path,
            super_animal= superanimal_name,
            model_name='hrnet_w32',
            detector_name='fasterrcnn_resnet50_fpn_v2',
            with_decoder=False
)


## delete `training-datasets` folder
if os.path.exists(os.path.join(project_path, 'training-datasets')):
    shutil.rmtree(os.path.join(project_path, 'training-datasets'))

if os.path.exists(os.path.join(project_path, 'dlc-models-pytorch')):
    shutil.rmtree(os.path.join(project_path, 'dlc-models-pytorch'))
    
superanimal_transfer_learning_shuffle = 1
deeplabcut.create_training_dataset(config_path, Shuffles=[superanimal_transfer_learning_shuffle], weight_init=weight_init, net_type='hrnet_w32')


deeplabcut.train_network(
    config_path,
    shuffle=1,
    epochs=50,
    save_epochs=10,
    superanimal_name=superanimal_name,
    batch_size= 16,
    keepdeconvweights=False,
    device="cuda:0",
    superanimal_transfer_learning=True
    )

deeplabcut.evaluate_network(config_path, Shuffles=[superanimal_transfer_learning_shuffle],
                            plotting="individual", comparisonbodyparts='all', show_errors= True, 
                             per_keypoint_evaluation=True )


deeplabcut.extract_save_all_maps(config=config_path, device="cuda:0", all_paf_in_one=False, Indices=[0, 25,50,75,100,125,150,175,199])


