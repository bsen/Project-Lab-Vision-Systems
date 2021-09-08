import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '../')
from my_utils import device

class BasicBlock3d(nn.Module):
    """
    A basic residual block for 3d volume processing with dropout.
    """
    
    def __init__(self, channel_in, channel_out, dropout_p):
        super().__init__()
        
        self.dropout_p = dropout_p
        self.conv1_bn = nn.Sequential(
                            nn.Conv3d(channel_in, channel_out, kernel_size=3,
                                      padding=1, bias=False),
                            nn.BatchNorm3d(channel_out))
        self.conv2_bn = nn.Sequential(
                            nn.Conv3d(channel_out, channel_out, kernel_size=3,
                                      padding=1, bias=False),
                            nn.BatchNorm3d(channel_out))
        
        if channel_in != channel_out:
            self.identity = nn.Conv3d(channel_in, channel_out, 1,
                                      bias=False)
        else:
            self.identity = nn.Identity()
    
    
    def forward(self, x):
        identity = self.identity(x)
        x = self.conv1_bn(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(x)
        x = self.conv2_bn(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        return F.relu(identity + x)
    
    def set_dropout(self, dropout_p):
        self.dropout_p = dropout_p


class CostProcessing(nn.Module):
    """A module calculating the cost volume from the left and right feature vectors
    and then using residual blocks with 3D convolutions to process this.
    Gets two tensors of shape (B, 32, H/4, W/4) as input and outputs
    a tensor of shape (B, 1, Disp/4, H/4, W/4).
    Here, Disp = 192"""
    def __init__(self, channels, dropout_p):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        
        for i in range(len(channels)-1):
            self.res_blocks.append(BasicBlock3d(channels[i], channels[i+1], 
                                                dropout_p=dropout_p))

                                   
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
        for block in self.res_blocks:
            cost = block(cost)
        
        return cost
    
    def set_dropout(self, dropout_p):
        for block in self.res_blocks:
            block.set_dropout(dropout_p)