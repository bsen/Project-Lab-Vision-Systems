import torch.nn as nn
from .cost_processing import CostProcessing
from .feature_extraction import FeatureExtractor
from .regressor import Regressor


class OurNet(nn.Module):
    """The complete net we use for predicting the disparity of a stereo input.

    Input: two tensors of the shape (B, 3, H, W)

    Output: disparity map of shape (B, H, W)"""

    def __init__(self, layers_feat=8, layers_cost=4):
        super().__init__()
        self.feature_extraction = FeatureExtractor(layers_feat, 'avg')
        self.cost_processing = CostProcessing(layers_cost)
        self.regressor = Regressor()

    def forward(self, left, right):
        left_feats = self.feature_extraction(left)
        right_feats = self.feature_extraction(right)

        cost = self.cost_processing(left, right)
        disp = self.regressor(cost)
        return disp
