import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """The siamese feature extractor.
    Gets as input one tensor of shape (B, 3, H, W) and outputs a tensor of
    shape (B, 32, H/4, W/4).
    """
    def __init__(self, num_blocks, pool_type):
        super().__init__()
        assert pool_type in ['max', 'avg']

        if pool_type == 'max':
            self.pooling = nn.MaxPool2d(2)
        else:
            self.pooling = nn.AvgPool2d(2)

        self.first_cnn = nn.Conv2d(3, 32, 3, padding=1)

        self.cnn_blocks = nn.ModuleList()
        for i in range(num_blocks):
            next_block = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, padding=1))
            self.cnn_blocks.append(next_block)

    def forward(self, x):
        output = self.first_cnn(x)

        for i, block in enumerate(self.cnn_blocks):
            if i == len(self.cnn_blocks)//3 or i == 2*(len(self.cnn_blocks)//3):
                output = self.pooling(output)
            output = nn.functional.relu(block(output)+output)

        return output
