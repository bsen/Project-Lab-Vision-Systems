import torch

import sys
sys.path.insert(0, '../')
from my_utils import device


class smoothL1:

    def __init__(self, beta):
        self.beta = beta

    def __call__(self, target, prediction):
        """
        The smooth L1 regression loss.
        The parameters should be of size (B, H, W).
        Pixels where the target disparity is 0.0 are ignored.
        """
        mask = (target != 0.0)

        # the number of elements which are unequal to 0.0 in each batch:
        N_i = mask.sum(dim=[1,2])

        t_minus_p = torch.abs(target - prediction)
        L2 = torch.pow(t_minus_p, 2.0)/(2.0*self.beta)
        L1 = t_minus_p - 0.5*self.beta

        use_L2 = (t_minus_p <= self.beta)
        use_L1 = torch.logical_not(use_L2)

        L = (L2*(torch.logical_and(mask, use_L2))).sum(dim=[1,2])
        L += (L1*(torch.logical_and(mask, use_L1))).sum(dim=[1,2])

        return torch.mean(L/N_i)


def three_pixel_err(target, prediction):
    """
    The 3-pixel error.
    The parameters should be of size (B, H, W).
    Pixels where the target disparity is 0.0 are ignored.
    """
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
