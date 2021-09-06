import torch
import torch.nn as nn
import sys
sys.path.insert(0, '../')
from my_utils import device

def get_pad(x):
    return x//2

class CostProcessing(nn.Module):
    """A module calculating the cost volume from the left and right feature vectors
    and then using residual blocks with 3D convolutions to process this.
    Gets two tensors of shape (B, 32, H/4, W/4) as input and outputs
    a tensor of shape (B, 1, Disp/4, H/4, W/4).
    Here, Disp = 192"""
    def __init__(self, channels, kernel_sizes, dropout_p):
        super().__init__()
        assert len(channels)-1 == len(kernel_sizes)
        self.channels = channels

        self.conv3d_blocks = nn.ModuleList()
        for i in range(0, len(channels)-2):
            next_block = nn.Sequential(
                    nn.Conv3d(channels[i], channels[i+1], kernel_sizes[i],
                              padding=get_pad(kernel_sizes[i])),
                    nn.BatchNorm3d(channels[i+1]),
                    nn.Dropout(dropout_p),
                    nn.ReLU(),
                    nn.Conv3d(channels[i+1], channels[i+1], kernel_sizes[i],
                              padding=get_pad(kernel_sizes[i])),
                    nn.BatchNorm3d(channels[i+1])
                    )
            self.conv3d_blocks.append(next_block)
            
        self.last_conv = nn.Conv3d(channels[-2], channels[-1], kernel_sizes[-1],
                                   padding=get_pad(kernel_sizes[-1]))

    def forward(self, left, right):
        """
        Here left and right are from the form (B, 32, H/4, W/4).
        They are feature tensors.
        """
        B, C, H4, W4 = left.shape
        # calculate the cost volume
        cost = torch.Tensor(B, C*2, 192//4, H4, W4).to(device)
        for i in range(192//4):
            if (i==0):
                cost[:, :C, i, :, :] = left
                cost[:, C:, i, :, :] = right
            else:
                cost[:, :C, i, :, i:] = left[:,:,:,i:]
                cost[:, C:, i, :, i:] = right[:,:,:,:-i]

        # feeding the cost volume through the Conv3d network
        for i, block in enumerate(self.conv3d_blocks):
            if self.channels[i] == self.channels[i+1]:
                cost = block(cost) + cost
            else:
                cost = block(cost)
        cost = self.last_conv(cost)
        
        return cost
