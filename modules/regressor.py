import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '../')
from my_utils import device


class Regressor(nn.Module):
    """
    The disparity regression module.
    Input is a tensor of shape (B, 1, Disp/4, H/4, W/4).

    We first upsample to size (B, Disp, H, W).
    Then we do a channelwise softmax on this.
    Then we weight and sum over the channels to get a output of size (B, 1, H, W).

    Output is the disparity map of size (B, H, W)
    """

    def __init__(self, dropout_p):
        super().__init__()
        self.weights = torch.arange(192).view(1, 192, 1, 1).to(device)
        
        
    def forward(self, x):
        B, _, _, H4, W4 = x.shape
        Disp = 192
        H = H4*4
        W = W4*4

        # upsample
        cost = F.interpolate(x, size=[Disp, H, W], mode='trilinear')
        cost = torch.squeeze(cost, dim=1)

        # channelwise softmax
        cost = F.softmax(cost, dim=1)

        # weight & sum
        output = torch.sum(cost * self.weights, dim=1, keepdim=False)
        
        return output
