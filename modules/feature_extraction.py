import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureExtractor(nn.Module):
    """
    The siamese feature extractor.
    Gets as input one tensor of shape (B, 3, H, W) and outputs a tensor of
    shape (B, 32, H/4, W/4).
    """
    def __init__(self, channels, dropout_p):
        super().__init__()

        self.cnn_blocks = nn.ModuleList()
        self.identities = nn.ModuleList()
        for i in range(len(channels)-1):
            if i == len(channels)//2:
                stride=2
            else:
                stride=1
                
            next_block = [nn.Conv2d(channels[i], channels[i+1],
                                     kernel_size=3,
                                     padding=1,
                                     stride=stride, bias=False),
                           nn.BatchNorm2d(channels[i+1]),
                           nn.Dropout2d(p=dropout_p),
                           nn.ReLU(),
                           nn.Conv2d(channels[i+1], channels[i+1], 
                                     kernel_size=3,
                                     padding=1, bias=False),
                           nn.BatchNorm2d(channels[i+1]),
                           nn.Dropout2d(p=dropout_p)]
            
            if i == 0:
                next_block.append(nn.AvgPool2d(2))
                next_identity = nn.Conv2d(channels[i], channels[i+1], kernel_size=1, 
                                          stride=2, bias=False)
            else:
                if stride != 1 or channels[i] != channels[i+1]:
                    # 1x1 convolution
                    next_identity = nn.Conv2d(channels[i], channels[i+1], kernel_size=1, 
                                              stride=stride, bias=False)
                else:
                    next_identity = nn.Identity()
            self.cnn_blocks.append(nn.Sequential(*next_block))
            self.identities.append(next_identity)

    def forward(self, x):
        for block, identity in zip(self.cnn_blocks, self.identities):
            x = F.relu(block(x) + identity(x))

        return x
