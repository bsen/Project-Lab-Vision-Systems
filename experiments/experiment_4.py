import torch
from ray import tune
import torch.optim as optim

import sys
sys.path.insert(0, './')
from my_utils import device, set_random_seed, get_tune_checkpoint

import training.training as training
from dataloader.KITTIloader import *
from modules.our_net import OurNet


"""
Using population based training for this problem (https://deepmind.com/blog/article/population-based-training-neural-networks)
"""

# datasets for training and validation
kitti_val = KittiDataset('val')
kitti_val_loader = torch.utils.data.DataLoader(dataset=kitti_val, batch_size=1,
                                               shuffle=True)

kitti_train = KittiDataset('train')
kitti_loader = torch.utils.data.DataLoader(dataset=kitti_train,
                                           batch_size=5,
                                           shuffle=True)

# the layers we use
layers = {'feat': [3,32,64,64,128,128,256,256,32], 'cost': [64,32,32,32,32,1]}


# the function which trains the model and reports checkpoints given
# a set of hyperparameters
def train(config, checkpoint_dir=None):
    channel_fe = layers['feat']
    channel_cp = layers['cost']
    
    # build the model
    model = OurNet(channel_fe=channel_fe, channel_cp=channel_cp,
                   dropout_p=config['dropout_p']).to(device)
    
    # load the pretrained model
    pre_checkpoint = torch.load(pretrained_path)
    model.load_state_dict(pre_checkpoint['model_state_dict'])
    model.set_dropout(config['dropout_p'])
    
    # build the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # create a scheduler which does not update the learning rate
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000000])
    
    epoch=0
    if checkpoint_dir:
        states = get_tune_checkpoint(os.path.join(checkpoint_dir, "checkpoint"))
        model_state, optimizer_state, scheduler_state, epoch = states
        
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        
        # load the learning rate from the config
        for param_group in optimizer.param_groups:
            param_group["lr"] = config["lr"]

    
    # training
    training.train_model(model=model, optimizer=optimizer, scheduler=scheduler,
            train_loader=kitti_loader, init_epoch=epoch, num_epochs=200, log_dir=None,
            valid_loader=kitti_val_loader, savefile=None,
            mes_time=False, use_amp=True, show_graph=False,
            use_tune=True, tune_interval=5, warmup_lr=False)

# the space where initial samples are drawn from
config = {
        'lr': tune.loguniform(1e-5, 1e-2),
        'dropout_p': tune.uniform(0.0, 0.5)
        }


scheduler = PopulationBasedTraining(
        time_attr="epoch",
        metric='err',
        mode='min',
        perturbation_interval=20,
        hyperparam_mutations={
            'lr': tune.loguniform(7e-7, 1e-1),
            'dropout_p': tune.uniform(0.0, 0.5)
        })

analysis = tune.run(
        train,
        name="experiment_4",
        scheduler=scheduler,
        metric="err",
        mode="min",
        stop=stopper,
        checkpoint_score_attr="min-err",
        keep_checkpoints_num=20,
        num_samples=10,
        config=config)