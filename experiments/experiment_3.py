import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
import torch_warmup_lr
import os

import sys
sys.path.insert(0, './')
from my_utils import device, set_random_seed

import training.training as training
from dataloader.KITTIloader import *
from dataloader import SceneFlowLoader as DA
from dataloader import listflowfile as lt
from modules.our_net import OurNet

# datasets for training and validation
kitti_val = KittiDataset('val')
kitti_val_loader = torch.utils.data.DataLoader(dataset=kitti_val, batch_size=1,
                                               shuffle=True)

kitti_train = KittiDataset('train')

layers = {'feat': [3,32,64,128,128,256,256,512,32], 'cost': [64,32,32,32,32,32,32,32,1]}

pretrained_path = '/home/user/brank/project/Project-Lab-Vision-Systems/saved_models/deep_pretrain.pt'


# the configuration for the hyperparameter search
config = {
        'lr': tune.loguniform(3e-6, 5e-3),
        'batch_size': tune.choice([4,5]),
        'step_lr': tune.quniform(35, 90, q=1.0),
        'num_warmup': tune.quniform(4, 8, q=1),
        'dropout_p': tune.uniform(0.0, 0.5)
        }

# the function which trains the model and reports checkpoints given
# a set of hyperparameters
def train(config, checkpoint_dir=None):

    kitti_loader = torch.utils.data.DataLoader(dataset=kitti_train,
                                               batch_size=config['batch_size'],
                                               shuffle=True)
    channel_fe = layers['feat']
    channel_cp = layers['cost']
    
    # build the model
    model = OurNet(channel_fe=channel_fe, channel_cp=channel_cp,
                   dropout_p=config['dropout_p']).to(device)
    
    # load the pretrained model
    pre_checkpoint = torch.load(pretrained_path)
    model.load_state_dict(pre_checkpoint['model_state_dict'])
    model.set_dropout(config['dropout_p'])
    
    # build the optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, int(config['step_lr']))
    scheduler = torch_warmup_lr.WarmupLR(scheduler, init_lr=1e-13, 
                                         num_warmup=config['num_warmup'], 
                                         warmup_strategy='cos')

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # training
    training.train_model(model=model, optimizer=optimizer, scheduler=scheduler,
            train_loader=kitti_loader, num_epochs=180, log_dir=None,
            valid_loader=kitti_val_loader, savefile=None,
            mes_time=False, use_amp=True, show_graph=False,
            use_tune=True, warmup_lr=True)


# set custom random seed
set_random_seed(21)

# use early stopping
trial_stop = tune.stopper.TrialPlateauStopper(metric='loss', std=0.005, num_results=4, grace_period=6)

# do hyperparameter search
result = result = tune.run(
        train,
        name='experiment_3',
        stop=trial_stop,
        resources_per_trial={'cpu':20, 'gpu':1},
        config=config,
        num_samples=10,
        checkpoint_score_attr='min-err',
        )

# output best trial
best_trial = result.get_best_trial('err', 'min', 'all')
print(f'Best trial config: {best_trial.config}')
print(f'Best trial final validation loss: {best_trial.last_result["loss"]}')
print(f'Best trial final validation error: {best_trial.last_result["err"]}')