"""
Some functions which we often use and therefore put in a module.
"""

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch_warmup_lr


# Setting the computing device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_path = '/home/user/brank/project/Project-Lab-Vision-Systems' # './' #'drive/MyDrive/colab/'

def set_random_seed(random_seed=None):
    """Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = 13
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return


def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def load_tune_checkpoint(model, checkpoint, optimizer=None, scheduler=None):
    states = get_tune_checkpoint(checkpoint)
    if len(states)==2:
        state_model, state_optimizer = states
    elif len(states) == 3:
        state_model, state_optimizer, state_scheduler = states
    elif len(states) == 4:
        state_model, state_optimizer, state_scheduler, _ = states
    else:
        raise Exception()
            
    model.load_state_dict(state_model)
    
    if optimizer is not None:
        optimizer.load_state_dict(state_optimizer)
    
    if scheduler is not None:
        scheduler.load_state_dict(state_scheduler)
        
        
def get_tune_checkpoint(checkpoint):
    # somehow there is a bug because getattr in the WarmupLR goes in a infinite loop
    # when loading a stored state_dict.
    # (The WarmupLR module is not written by myself.)
    
    # So if you want to load a model from a checkpoint created by tune, please use this workaround
    def new_getattr(self, name):
        if name == '_scheduler':
            self._scheduler = None
            return self._scheduler
        return getattr(self._scheduler, name)
    
    old_getattr = torch_warmup_lr.WarmupLR.__getattr__
    torch_warmup_lr.WarmupLR.__getattr__ = new_getattr
    
    states = torch.load(checkpoint, device)
    
    torch_warmup_lr.WarmupLR.__getattr__ = old_getattr

    return states
