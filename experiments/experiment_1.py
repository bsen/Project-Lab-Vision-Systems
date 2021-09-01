import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune

import sys
sys.path.insert(0, '../')
from my_utils import device, set_random_seed

import training.training as training
from dataloader.KITTIloader import *
from dataloader import SceneFlowLoader as DA
from dataloader import listflowfile as lt
import modules

# datasets for training and validation
kitti_val = KittiDataset('val')
kitti_val_loader = torch.utils.data.DataLoader(dataset=kitti_val, batch_size=1,
                                               shuffle=True)
kitti_train = KittiDataset('train')

# datasets for pretraining
driv_left_img, driv_right_img, driv_left_disp, driv_right_disp,_,_,_,_ =\
        lt.dataloader('datasets/scene_flow/', 'driving')
pre_set = DA.myImageFolder(driv_left_img, driv_right_img, driv_left_disp, False, driv_right_disp)
pre_loader = torch.utils.data.DataLoader(dataset=pre_set, batch_size=30, shuffle=True)

# the configuration for the hyperparameter search
config = {
        'lr': tune.loguniform(1e-6, 3e-3),
        'layers_feat': tune.quniform(4,10,q=1.0),
        'layers_cost': tune.quniform(2,7,q=1.0),
        'kernel_feat': tune.choice([3,5,7]),
        'kernel_cost': tune.choice([3,5,7]),
        'batch_size': tune.quniform(5, 20, q=1.0),
        'optimizer': tune.choice(['adam', 'sgd']),
        'step_lr': tune.quniform(15, 60, q=1.0)
        }

# the function which trains the model and reports checkpoints given
# a set of hyperparameters
def train(config, checkpoint_dir=None):
    # calculate the parameters for the feature extractor
    channel_fe = [3]
    channel_fe_16 = int(config['layers_feat'])//2
    channel_fe_32 = int(config['layers_feat'])//2 + int(config['layers_feat'])%2
    channel_fe = channel_fe + [16]*channel_fe_16
    channel_fe = channel_fe + [32]*channel_fe_32
    kernel_fe = [int(config['kernel_feat'])]*(int(config['layers_feat']))

    # calculate the parameters the cost processing
    channel_cp_64 = int(config['layers_cost'])//2 + int(config['layers_cost'])%2
    channel_cp_32 = int(config['layers_cost'])//2
    channel_cp = [64] * channel_cp_64
    channel_cp = channel_cp + [32] * channel_cp_32
    channel_cp.append(1)
    kernel_cp = [int(config['kernel_cost'])]*int(config['layers_cost'])

    model = modules.our_net(channel_fe=channel_fe, kernel_fe=kernel_fe,
            channel_cp=channel_cp, kernel_cp=kernel_cp)

    kitti_loader = torch.utils.data.DataLoader(dataset=kitti_train,
                                               batch_size=int(config['batch_size']),
                                               shuffle=True)
    # build the optimizers and schedulers
    if config['optimizer'] == 'adam':
        pre_optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'sgd':
        pre_optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    pre_scheduler = optim.lr_scheduler.StepLR(pre_optimizer, int(config['step_lr']))

    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, int(config['step_lr']))

    # training
    training.train_model(model=model, optimizer=optimizer, scheduler=scheduler,
            train_loader=kitti_loader, num_epochs=3, log_dir=None,
            valid_loader=kitti_val_loader, savefile=None,
            mes_time=False, pretrain_optimizer=pre_optimizer,
            pretrain_scheduler=pre_scheduler, pretrain_loader=pre_loader,
            pretrain_epochs=3, use_amp=True, show_graph=False,
            tune_checkpoint_dir=checkpoint_dir)


# set custom random seed
set_random_seed(42)

# do hyperparameter search
result = tune.run(
        train,
        name='experiment 1',
        resources_per_trial={'cpu':12, 'gpu':1},
        config=config,
        num_samples=2,
        checkpoint_score_attr='min-err',
        )

