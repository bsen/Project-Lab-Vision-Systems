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
from my_utils import load_tune_checkpoint, get_tune_checkpoint
from my_utils import set_random_seed
from training import training


device = 'cuda'
set_random_seed(22)

# dataloaders

kitti_val_loader = torch.utils.data.DataLoader(dataset=KITTIloader.KittiDataset('val'),
                                               batch_size=1,
                                               shuffle=True)


# the channels of experiment 2
layers = {'cost_heavy': {'feat': [3,16,32,64,128,256,32], 'cost': [64, 32, 32, 32, 32, 1]}, 
          'feat_heavy': {'feat':[3,16,64,64,128,128,128,256,32], 'cost': [64, 32, 32, 32, 1]}}

# The best configuration of experiment 3
best_config = {
    'lr': 0.0011789840332586685,
    'layers': 'cost_heavy',
    'batch_size': 5,
    'step_lr': 63,
    'num_warmup': 5,
    'dropout_p': 0.01858016508799909
    }

# The best checkpoint of experiment 3:
best_checkpoint = '/home/user/brank/ray_results/experiment_2/train_2e5f0ed2_5_batch_size=5,dropout_p=0.01858,layers=cost_heavy,lr=0.001179,num_warmup=6.0,step_lr=63.0_2021-09-10_15-38-14/checkpoint_000075/checkpoint'

def train(config, checkpoint_dir=None):
    channel_fe=layers[best_config['layers']]['feat']
    channel_cp=layers[best_config['layers']]['cost']
    
    # build the model
    model = OurNet(channel_fe=channel_fe, channel_cp=channel_cp,
                   dropout_p=best_config['dropout_p']).to(device)

    kitti_train = torch.utils.data.DataLoader(dataset=KITTIloader.KittiDataset('train'), 
                                                        batch_size=best_config['batch_size'], 
                                                        shuffle=True)
    
    # build optimizer
    optimizer = optim.Adam(model.parameters(), lr=best_config['lr'])

    # load checkpoint
    load_tune_checkpoint(model=model, checkpoint=best_checkpoint, optimizer=optimizer)

    # build scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 150])

    epoch=0
    if checkpoint_dir:
        states = get_tune_checkpoint(
            os.path.join(checkpoint_dir, "checkpoint"))
        model_state, optimizer_state, scheduler_state, epoch = states
        
        scheduler_state.load_state_dict(scheduler_state)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # fine tune further
    losses = training.train_model(model=model, optimizer=optimizer, scheduler=scheduler,
            train_loader=kitti_train, num_epochs=200,
            log_dir=None, init_epoch=epoch,
            valid_loader=kitti_val_loader, savefile=None,
            mes_time=False, use_amp=True, show_graph=False,
            use_tune=True, warmup_lr=False)
    torch.save(losses, '/home/user/brank/ray_results/experiment_2_finetune/losses.pt')

    
# run the training
result = result = tune.run(
        train,
        name='experiment_2_finetune',
        resources_per_trial={'cpu':20, 'gpu':1},
        config={},
        num_samples=1,
        checkpoint_score_attr='min-err',
        )
