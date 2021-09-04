import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../')
from my_utils import device

from dataloader.KITTIloader import unnormalize


class smoothL1:
    def __init__(self, beta):
        self.nnSmoothL1 = nn.SmoothL1Loss(beta=beta)

    def __call__(self, target, prediction):
        mask = (target > 0.0)
        mask.detach_()
        return self.nnSmoothL1(prediction[mask], target[mask])


def three_pixel_err(target, prediction):
    """
    The 3-pixel error.
    The parameters should be of size (B, H, W).
    Pixels where the target disparity is 0.0 are ignored.
    """
    target = unnormalize(target)
    prediction = unnormalize(prediction)
    
    B, _, _ = target.shape

    t_minus_p = torch.abs(target - prediction)
    lower_3 = (t_minus_p < 3)
    lower_005_x = t_minus_p < (0.05 * target)
    good_pred = torch.logical_or(lower_3, lower_005_x)

    mask = (target != 0.0)
    # the number of elements which are unequal to 0.0 in each batch:
    N_i = mask.sum(dim=[1,2])

    # the number of elements where the prediction is good and which are
    # unequal to 0.0 in each batch:
    good_N_i = torch.logical_and(good_pred, mask).sum(dim=[1,2])

    pe3 = torch.ones(B, device=device) - good_N_i/N_i
    return pe3.mean()
