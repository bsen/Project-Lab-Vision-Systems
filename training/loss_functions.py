import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../')
from my_utils import device


class smoothL1:
    def __init__(self, beta):
        self.nnSmoothL1 = nn.SmoothL1Loss(beta=beta)

    def __call__(self, target, prediction):
        mask = (target != 0.0)
        mask.detach_()
        return self.nnSmoothL1(prediction[mask], target[mask])

    
@torch.no_grad()
def three_pixel_err(target, prediction):
    """
    The 3-pixel error.
    The parameters should be of size (B, H, W).
    Pixels where the target disparity is 0.0 are ignored.
    """
    
    mask = torch.logical_and((target != 0.0), (target<=192.0))
    
    target = target[mask]
    prediction = prediction[mask]
    
    t_minus_p = torch.abs(target - prediction)
    lower_3 = (t_minus_p < 3)
    lower_005_x = t_minus_p < (0.05 * target)
    good_pred = torch.logical_or(lower_3, lower_005_x)

    N = torch.numel(target)

    # the number of elements where the prediction is good:
    N_good = good_pred.sum()

    return (1 - (N_good/N))
    
