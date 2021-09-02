import torch
import torch.nn as nn
import math
import torchvision.models as models


def get_pad(x):
    return x//2

class ResnetRegressor(nn.Module):
    """
    A siamese feature extractor which uses part of pretrained resnet18 as a fixed feature extractor.
    Gets as input one tensor of shape (B, 3, H, W) and outputs a tensor of
    shape (B, 32, H/4, W/4).
    """
    def __init__(self, fixed=False):
        super().__init__()
        self.fixed = fixed
        resnet = models.resnet18(pretrained=True)
        
        for param in resnet.parameters():
            if fixed:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        
        if self.fixed:
            self.transconv = nn.Conv2d(128, 32, 1)
        else:
            self.transconv = nn.ConvTranspose2d(128, 32, kernel_size=3, 
                                                stride=2, padding=1, 
                                                output_padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.fixed:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.transconv(x)
        return x
