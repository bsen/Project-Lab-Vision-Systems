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

# the configuration for the hyperparameter search
config = {
        'lr': tune.loguniform(5e-8, 3e-3),
        'layers_feat': tune.choice([[3,16,16,16,16,16,16,16,32], [3,16,16,16,32,32,32,32],
                                   [3,32,32,32,32,32]]),
        'layers_cost': tune.choice([[64, 32, 32, 32, 1], [64, 32, 16, 16, 16, 16, 1], [64, 32, 32, 16, 16, 1]]),
        'batch_size': tune.quniform(4, 6, q=1.0),
        'optimizer': tune.choice(['adam', 'sgd']),
        'step_lr': tune.quniform(35, 90, q=1.0),
        'num_warmup': tune.quniform(4, 10, q=1),
        'dropout_p': tune.uniform(0.01, 0.7),
        'normalizing_factor': tune.uniform(150.0, 500.0)
        }

# the function which trains the model and reports checkpoints given
# a set of hyperparameters
def train(config, checkpoint_dir=None):
    
    kernel_fe = [3]*(len(config['layers_feat'])-1)
    kernel_cp = [3]*(len(config['layers_cost'])-1)

    model = OurNet(channel_fe=config['layers_feat'], kernel_fe=kernel_fe,
                   channel_cp=config['layers_cost'], kernel_cp=kernel_cp,
                   normalizing_factor=config['normalizing_factor']).to(device)

        
    kitti_loader = torch.utils.data.DataLoader(dataset=kitti_train,
                                               batch_size=int(config['batch_size']),
                                               shuffle=True)
    # build the optimizers and schedulers
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, int(config['step_lr']))
    scheduler = torch_warmup_lr.WarmupLR(scheduler, init_lr=1e-12, 
                                         num_warmup=config['num_warmup'], 
                                         warmup_strategy='cos')

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # training
    training.train_model(model=model, optimizer=optimizer, scheduler=scheduler,
            train_loader=kitti_loader, num_epochs=4, log_dir=None,
            valid_loader=kitti_val_loader, savefile=None,
            mes_time=False, use_amp=False, show_graph=False,
            use_tune=True, sched_before_optim=True)


# set custom random seed
set_random_seed(21)

# use early stopping
trial_stop = tune.stopper.TrialPlateauStopper(metric='loss', std=0.005, num_results=4, grace_period=20)

# do hyperparameter search
result = result = tune.run(
        train,
        name='experiment_1',
        stop=trial_stop,
        resources_per_trial={'cpu':12, 'gpu':1},
        config=config,
        num_samples=3,
        checkpoint_score_attr='min-err',
        )

# output best trial
best_trial = result.get_best_trial('err', 'min', 'all')
print(f'Best trial config: {best_trial.config}')
print(f'Best trial final validation loss: {best_trial.last_result["loss"]}')
print(f'Best trial final validation error: {best_trial.last_result["err"]}')