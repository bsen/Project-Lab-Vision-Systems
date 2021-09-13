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


def load_tune_checkpoint(model, checkpoint, optimizer=None):
    # somehow there is a bug because getattr in the WarmupLR sometimes goes in a infinite loop.
    # (The WarmupLR module is not written by myself.)
    # This was not an issue before but here it is.
    # Because I don't need the scheduler in this notebook, I just set it to a new method
    # and by that the we don't get a "Maximum recursion depth exceeded" error.

    # So if one wants to load a model from a checkpoint created by tune, please use this workaround
    def new_getattr(self, name):
        if name == '_scheduler':
            self._scheduler = None
            return self._scheduler
        return getattr(self._scheduler, name)
    
    old_getattr = torch_warmup_lr.WarmupLR.__getattr__
    torch_warmup_lr.WarmupLR.__getattr__ = new_getattr
    
    state_model, state_optimizer, state_scheduler = torch.load(checkpoint, device)
    model.load_state_dict(state_model)
    
    torch_warmup_lr.WarmupLR.__getattr__ = old_getattr
    
    if optimizer is not None:
        optimizer.load_state_dict(state_optimizer)