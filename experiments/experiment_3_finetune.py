import torch
import torch.optim as optim
import numpy as np
from ray import tune
import os
import sys

sys.path.insert(0, './')
from dataloader import KITTIloader, listflowfile
import dataloader.SceneFlowLoader as SFL
from modules.our_net import OurNet
from my_utils import load_tune_checkpoint
from my_utils import set_random_seed
from training import training


device = 'cuda'
set_random_seed(22)

# dataloaders
kitti_train = torch.utils.data.DataLoader(dataset=KITTIloader.KittiDataset('train'), 
                                                        batch_size=5, 
                                                        shuffle=True)

kitti_val_loader = torch.utils.data.DataLoader(dataset=KITTIloader.KittiDataset('val'),
                                               batch_size=1,
                                               shuffle=True)


# the channels of experiment 3
layers = {'feat': [3,32,64,128,128,256,256,512,32], 'cost': [64,32,32,32,32,32,32,32,1]}

# The best configuration of experiment 3
best_config = {'dropout_p': 0.21800533478896061,
          'lr': 0.0004859878519547141,
          'num_warmup': 7,
          'step_lr': 55}

# The best checkpoint of experiment 3:
best_checkpoint = '/home/user/brank/ray_results/experiment_3/train_1748b_00008_8_dropout_p=0.21801,lr=0.00048599,num_warmup=7.0,step_lr=55.0_2021-09-10_05-16-01/checkpoint_000075/checkpoint'

channel_fe = layers['feat']
channel_cp = layers['cost']


def train(config, checkpoint_dir=None):
    # build the model
    model = OurNet(channel_fe=channel_fe, channel_cp=channel_cp,
                   dropout_p=best_config['dropout_p']).to(device)

    # build optimizer
    optimizer = optim.Adam(model.parameters(), lr=best_config['lr'])

    # load checkpoint
    load_tune_checkpoint(model=model, checkpoint=best_checkpoint, optimizer=optimizer)

    # build scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, best_config['step_lr'])



    if checkpoint_dir:
        states = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model_state, optimizer_state, scheduler_state = states
        
        scheduler_state.load_state_dict(scheduler_state)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # fine tune further
    training.train_model(model=model, optimizer=optimizer, scheduler=scheduler,
            train_loader=kitti_train, num_epochs=200,
            log_dir=None,
            valid_loader=kitti_val_loader, savefile=None,
            mes_time=False, use_amp=True, show_graph=False,
            use_tune=True, warmup_lr=False)

    
# run the training
result = result = tune.run(
        train,
        name='experiment_3_finetune',
        resources_per_trial={'cpu':20, 'gpu':1},
        config={},
        num_samples=1,
        checkpoint_score_attr='min-err',
        )
