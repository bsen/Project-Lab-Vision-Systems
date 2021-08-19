import torch
import torch.nn as nn
import sys
sys.path.insert(0, '../')
from my_utils import device


class CostProcessing(nn.Module):
    """A module calculating the cost volume from the left and right feature vectors
    and then using residual blocks with 3D convolutions to process this.
    Gets two tensors of shape (B, 32, H/4, W/4) as input and outputs
    a tensor of shape (B, 1, Disp/4, H/4, W/4).
    Here, Disp = 192"""
    def __init__(self, num_blocks):
        super().__init__()

        self.first_block = nn.Sequential(
                nn.Conv3d(64, 32, 3, padding=1),
                nn.ReLU()
        )
        self.conv3d_blocks = nn.ModuleList()
        for i in range(num_blocks):
            next_block = nn.Sequential(
                    nn.Conv3d(32, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(32, 32, 3, padding=1)
                    # here we could add a ReLU
                    )
            self.conv3d_blocks.append(next_block)

        self.last_block = nn.Sequential(
                nn.Conv3d(32, 1, 3, padding=1),
                nn.ReLU()
        )

    def forward(self, left, right):
        """Here left and right are from the form (B, 32, H/4, W/4).
        They are feature tensors."""
        B, C, H4, W4 = left.shape
        cost = torch.Tensor(B, C*2, 192//4, H4, W4).to(device)
        for i in range(192//4):
            if (i==0):
                cost[:, :C, i, :, :] = left
                cost[:, C:, i, :, :] = right
            else:
                cost[:, :C, i, :, i:] = left[:,:,:,i:]
                cost[:, C:, i, :, i:] = right[:,:,:,:-i]

        output = self.first_block(cost)
        for block in self.conv3d_blocks:
            output = block(output) + output

        output = self.last_block(output)

        return output
