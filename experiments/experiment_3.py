import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ray import tune
import torch_warmup_lr
import os
from collections import defaultdict, deque

import sys
sys.path.insert(0, './')
from my_utils import device, set_random_seed

import training.training as training
from dataloader.KITTIloader import *
from dataloader import SceneFlowLoader as DA
from dataloader import listflowfile as lt
from modules.our_net import OurNet

"""
A module performing a random search for hyperparameters using the raytune framework.
"""

# datasets for training and validation
kitti_val = KittiDataset('val')
kitti_val_loader = torch.utils.data.DataLoader(dataset=kitti_val, batch_size=1,
                                               shuffle=True)

kitti_train = KittiDataset('train')

layers = {'feat': [3,32,64,128,128,256,256,512,32], 'cost': [64,32,32,32,32,32,32,32,1]}

pretrained_path = '/home/user/brank/project/Project-Lab-Vision-Systems/saved_models/deep_pretrain.pt'


# the configuration for the hyperparameter search
config = {
        'lr': tune.loguniform(3e-5, 5e-3),
        'step_lr': tune.quniform(45, 90, q=1.0),
        'num_warmup': tune.quniform(4, 8, q=1),
        'dropout_p': tune.uniform(0.0, 0.3)
        }

# the function which trains the model and reports checkpoints given
# a set of hyperparameters
def train(config, checkpoint_dir=None):

    kitti_loader = torch.utils.data.DataLoader(dataset=kitti_train,
                                               batch_size=5,
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
        states = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model_state, optimizer_state, scheduler_state = states
        
        scheduler_state.load_state_dict(scheduler_state)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # training
    training.train_model(model=model, optimizer=optimizer, scheduler=scheduler,
            train_loader=kitti_loader, num_epochs=100, log_dir=None,
            valid_loader=kitti_val_loader, savefile=None,
            mes_time=False, use_amp=True, show_graph=False,
            use_tune=True, warmup_lr=True)


# set custom random seed
set_random_seed(21)

# build a custom stopper which early stops single trials

class CustomStopper(tune.Stopper):
    """
    Some parts of the code of this class are copied from the TrialPlateauStopper
    code from raytune.
    """
    
    def __init__(self, std, grace_period, num_results):
        self._std = std
        self._grace_period = grace_period
        self._num_results = num_results
        
        self._iter = defaultdict(lambda: 0)
        self._trial_results = defaultdict(
                    lambda: deque(maxlen=self._num_results))
        self._errors = defaultdict(
                    lambda: deque(maxlen=self._num_results))

    def __call__(self, trial_id, result):
        """
        Gets as input the current trial and stops it when the function was called more
        than grace_period amount of times for this trial_id and
        one of the following criteria are met:
        - the standard deviation on the last num_results is lower than std (for this trial_id)
        - the function was called more than 14 times (we are in epoch >=40) and
            the error is still higher than 0.2
        - the function was called more than 20 times (we are in epoch >=58) and
            the error is still higher than 0.1
        - the function was called more than 33 times and the mean error of the
            last num_results results is higher than
            0.085
        - the loss is nan or infinity (this happend to me in the past
                                       due to using automatic mixed precision)
        """
        loss = result.get('loss')
        err = result.get('err')
        
        self._trial_results[trial_id].append(loss)
        self._errors[trial_id].append(err)
        self._iter[trial_id] += 1
        
        # If still in grace period, do not stop yet
        if self._iter[trial_id] < self._grace_period:
            return False

        if (self._iter[trial_id] >= 14) and (err >= 0.2):
            return True
        
        if (self._iter[trial_id] >= 20) and (err >= 0.1):
            return True
        
        if np.isnan(loss):
            print('The loss is nan')
            return True
        
        if np.isinf(loss):
            print('The loss is inf: loss =', loss)
            return True
        
        try:
            if (self._iter[trial_id] >= 30) and (np.mean(self._errors[trial_id]) >= 0.085):
                return True
        except Exception:
            pass
        
        # If not enough results yet, do not stop yet
        if len(self._trial_results[trial_id]) < self._num_results:
            return False
        
        # Calculate stdev of last `num_results` results
        try:
            current_std = np.std(self._trial_results[trial_id])
        except Exception:
            current_std = float("inf")

        # If stdev is lower than threshold, stop early.
        return current_std < self._std


    def stop_all(self):
        return False

# use early stopping
trial_stop = CustomStopper(std=0.015, grace_period=4, num_results=4)


# do hyperparameter search
result = tune.run(
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