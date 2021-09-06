import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '../')
from my_utils import device


class CostProcessing(nn.Module):
    """A module calculating the cost volume from the left and right feature vectors
    and then using residual blocks with 3D convolutions to process this.
    Gets two tensors of shape (B, 32, H/4, W/4) as input and outputs
    a tensor of shape (B, 1, Disp/4, H/4, W/4).
    Here, Disp = 192"""
    def __init__(self, channels, dropout_p):
        super().__init__()

        self.conv3d_blocks = nn.ModuleList()
        self.identities = nn.ModuleList()
        
        for i in range(len(channels)-1):
            next_block = nn.Sequential(
                    nn.Conv3d(channels[i], channels[i+1], kernel_size=3,
                              padding=1, bias=False),
                    nn.BatchNorm3d(channels[i+1]),
                    nn.Dropout(dropout_p),
                    nn.ReLU(),
                    nn.Conv3d(channels[i+1], channels[i+1], kernel_size=3,
                              padding=1, bias=False),
                    nn.BatchNorm3d(channels[i+1]),
                    nn.Dropout(dropout_p)
                    )
            self.conv3d_blocks.append(next_block)
            
            if channels[i] != channels[i+1]:
                next_identity = nn.Conv3d(channels[i], channels[i+1], 1,
                                          bias=False)
            else:
                next_identity = nn.Identity()
                
            self.identities.append(next_identity)
            

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
        for identity, block in zip(self.identities, self.conv3d_blocks):
            cost = F.relu(block(cost) + identity(cost))
        
        return cost
