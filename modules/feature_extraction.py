import torch
import torch.nn as nn
import math


def get_pad(x):
    return x//2

class FeatureExtractor(nn.Module):
    """
    The siamese feature extractor.
    Gets as input one tensor of shape (B, 3, H, W) and outputs a tensor of
    shape (B, 32, H/4, W/4).
    """
    def __init__(self, channels=[3,4,4,8,8,8,16,16,32],
                 kernel_sizes=[3,3,3,3,3,3,3,3]):
        super().__init__()
        assert len(channels)-1 == len(kernel_sizes)

        self.channels = channels
        self.first_cnn = nn.Sequential(
                    nn.Conv2d(channels[0], channels[1], kernel_sizes[0],
                                   padding=get_pad(kernel_sizes[0])),
                    nn.AvgPool2d(2))

        self.cnn_blocks = nn.ModuleList()
        for i in range(1, len(channels)-1):
            if i == len(channels)//2:
                stride=2
            else:
                stride=1
            next_block = nn.Sequential(nn.Conv2d(channels[i], channels[i+1],
                                                 kernel_sizes[i],
                                                 padding=get_pad(kernel_sizes[i]),
                                                 stride=stride),
                                       nn.ReLU(),
                                       nn.Conv2d(channels[i+1], channels[i+1],
                                                 kernel_sizes[i],
                                                 padding=get_pad(kernel_sizes[i]))
                                       )
            self.cnn_blocks.append(next_block)

    def forward(self, x):
        output = self.first_cnn(x)

        for i, block in enumerate(self.cnn_blocks, start=1):
            if (self.channels[i] == self.channels[i+1]) and (block[0].stride == (1,1)):
                output = nn.functional.relu(block(output)+output)
            else:
                output = nn.functional.relu(block(output))

        return output
