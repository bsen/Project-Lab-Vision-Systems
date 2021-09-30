import torch
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
import torch.optim as optim

import sys
sys.path.insert(0, './')
from my_utils import device, set_random_seed, get_tune_checkpoint
import training.training as training
from dataloader.KITTIloader import *
from modules.our_net import OurNet


"""
Using population based training for this problem + SGD optimizer
(https://deepmind.com/blog/article/population-based-training-neural-networks)
"""

# set random seed
set_random_seed(10)

# datasets for training and validation
kitti_val = KittiDataset('val')
kitti_val_loader = torch.utils.data.DataLoader(dataset=kitti_val, batch_size=1,
                                               shuffle=True)

kitti_train = KittiDataset('train')
kitti_loader = torch.utils.data.DataLoader(dataset=kitti_train,
                                           batch_size=5,
                                           shuffle=True)

# the layers we use
layers = {'feat': [3,32,64,64,128,128,256,256,32], 'cost': [64,32,32,32,32,32,1]}


# the function which trains the model and reports checkpoints given
# a set of hyperparameters
def train(config, checkpoint_dir=None):
    channel_fe = layers['feat']
    channel_cp = layers['cost']
    
    # build the model
    model = OurNet(channel_fe=channel_fe, channel_cp=channel_cp,
                   dropout_p=config['dropout_p']).to(device)
    
    # build the optimizer
    #optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    
    # create a scheduler which does not update the learning rate
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000000])
    
    epoch=0
    # load checkpoint_dir if provided
    if checkpoint_dir:
        states = get_tune_checkpoint(os.path.join(checkpoint_dir, "checkpoint"))
        model_state, optimizer_state, scheduler_state, epoch = states
       
        # go to next epoch
        epoch += 1 

        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

        # load the learning rate from the config
        for param_group in optimizer.param_groups:
            param_group["lr"] = config["lr"]
            param_group['momentum'] = config['momentum']
            
    model.set_dropout(config['dropout_p'])
    
    # training
    training.train_model(model=model, optimizer=optimizer, scheduler=scheduler,
            train_loader=kitti_loader, init_epoch=epoch, num_epochs=271,
                         log_dir=None,
                         valid_loader=kitti_val_loader, savefile=None,
                         mes_time=False, use_amp=True, show_graph=False,
                         use_tune=True, tune_interval= 5, 
                         warmup_lr=False)

# the space where initial samples are drawn from
config = {
        'lr': tune.loguniform(1e-5, 1e-2),
        'dropout_p': tune.uniform(0.0, 0.5),
        'momentum': tune.uniform(0.8, 0.999)
        }

# the Population based training scheduler, defines how 
# and when to pertube. (documentation:
# https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#population-based-training-tune-schedulers-populationbasedtraining )
scheduler = PopulationBasedTraining(
        time_attr="epoch",
        perturbation_interval= 30,
        hyperparam_mutations={
            'lr': tune.loguniform(5e-7, 5e-2),
            'dropout_p': tune.uniform(0.0, 0.6),
            'momentum': tune.uniform(0.0, 0.999)
        })

# run the experiment
result = tune.run(
        train,
        name="experiment_5",
        resources_per_trial={'cpu':20, 'gpu':1},
        scheduler=scheduler,
        metric="err",
        mode="min",
        checkpoint_score_attr="min-err",
        keep_checkpoints_num=20,
        num_samples=8,
        config=config)

# output best trial
best_trial = result.get_best_trial('err', 'min', 'all')
print(f'Best trial final validation loss: {best_trial.last_result["loss"]}')
print(f'Best trial final validation error: {best_trial.last_result["err"]}')
print(f'Best trial config: {best_trial.config}')
