import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicBlock(nn.Module):
    """
    A basic residual block with dropout.
    """
    
    def __init__(self, channel_in, channel_out, stride, dropout_p, pool=False):
        super().__init__()
        assert (not pool) or (stride == 1)
        
        self.dropout_p = dropout_p
        self.conv1_bn = nn.Sequential(
                            nn.Conv2d(channel_in, channel_out,
                                        kernel_size=3,
                                        padding=1,
                                        stride=stride, bias=False),
                            nn.BatchNorm2d(channel_out))

        self.conv2_bn = nn.Sequential(
                            nn.Conv2d(channel_out, channel_out, 
                                        kernel_size=3,
                                        padding=1, bias=False),
                            nn.BatchNorm2d(channel_out))
        if pool:
            self.pool = nn.AvgPool2d(2)
        else:
            self.pool = nn.Identity()
        
        if pool:
            self.identity = nn.Conv2d(channel_in, channel_out, kernel_size=1, 
                                      stride=2, bias=False)
        else:
            if stride != 1 or (channel_in != channel_out):
                # 1x1 convolution
                self.identity = nn.Conv2d(channel_in, channel_out, kernel_size=1, 
                                          stride=stride, bias=False)
            else:
                self.identity = nn.Identity()
                
    def forward(self, x):
        identity = self.identity(x)
        x = self.conv1_bn(x)
        x = F.dropout2d(x, p=self.dropout_p, training=self.training)
        x = F.relu(x)
        x = self.conv2_bn(x)
        x = F.dropout2d(x, p=self.dropout_p, training=self.training)
        x = self.pool(x)
        
        return F.relu(identity + x)
    
    def set_dropout(self, dropout_p):
        self.dropout = dropout_p
        

class FeatureExtractor(nn.Module):
    """
    The siamese feature extractor.
    Gets as input one tensor of shape (B, 3, H, W) and outputs a tensor of
    shape (B, 32, H/4, W/4).
    """
    def __init__(self, channels, dropout_p):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            if i == len(channels)//2:
                stride=2
            else:
                stride=1
            if i == 0:
                pool = True
            else:
                pool = False
                
            self.res_blocks.append(BasicBlock(channels[i], channels[i+1], stride=stride, 
                                              dropout_p=dropout_p, pool=pool))

    def forward(self, x):
        for block in self.res_blocks:
            x = block(x)
        return x
    
    def set_dropout(self, dropout_p):
        for block in self.res_blocks:
            block.set_dropout(dropout_p)
