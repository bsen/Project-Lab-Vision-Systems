import torch
import torch.nn as nn
import torch.optim as optim
import torch_warmup_lr
import os
import gc

import sys
sys.path.insert(0, './')
from my_utils import device, set_random_seed
from dataloader import KITTIloader
from dataloader import listflowfile
import dataloader.SceneFlowLoader as SFL
from training import training
from modules.our_net import OurNet

"""
This python-module is used to pretrain the model on Scene Flow driving dataset for 
experiment 3.
"""

# set custom random seed
set_random_seed(100)

# create kitti validation loader
kitti_val = KITTIloader.KittiDataset('val')
kitti_val_loader = torch.utils.data.DataLoader(dataset=kitti_val, batch_size=1,
                                               shuffle=True)
# create loader for the Scene Flow driving dataset
left_img, right_img, left_disp, right_disp, _, _, _, _ = \
    listflowfile.dataloader('/home/user/brank/project/Project-Lab-Vision-Systems/datasets/scene_flow/', 'driving')
driving_train = SFL.myImageFolder(left=left_img, right=right_img, left_disparity=left_disp, 
                                  training=True, right_disparity=right_disp)

driving_loader = torch.utils.data.DataLoader(dataset=driving_train,
                                             batch_size=5,
                                             shuffle=True)

# the layer-configurations for the hyperparameter search
layers = {'feat': [3,32,64,128,128,256,256,512,32], 'cost': [64,32,32,32,32,32,32,32,1]}

channel_fe = layers['feat']
channel_cp = layers['cost']

model = OurNet(channel_fe=channel_fe, channel_cp=channel_cp,
               dropout_p=0.2).to(device)

# build the optimizers and schedulers
optimizer = optim.Adam(model.parameters(), lr=3e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, 40)
scheduler = torch_warmup_lr.WarmupLR(scheduler, init_lr=1e-13, 
                                     num_warmup=3, 
                                     warmup_strategy='cos')
model_type = 'deep_pretrain'
# training
# The low number of epochs is due to the fact that the number of elements in
# the Scene Flow driving dataset is high.
training.train_model(model=model, optimizer=optimizer, scheduler=scheduler,
        train_loader=driving_loader, num_epochs=4,
        log_dir=model_type,
        valid_loader=kitti_val_loader, savefile='saved_models/'+model_type+'.pt',
        mes_time=False, use_amp=True, show_graph=False,
        use_tune=False, warmup_lr=True, sched_iters=25)