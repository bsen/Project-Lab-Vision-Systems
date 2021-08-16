"""
Some functions which we often use and therefore put in a module.
"""

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

# Setting the computing device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_path = './' #'drive/MyDrive/colab/'

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
