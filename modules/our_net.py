import torch.nn as nn
from .cost_processing import CostProcessing
from .feature_extraction import FeatureExtractor
from .regressor import Regressor
from .resnet_extraction import ResnetFE


class OurNet(nn.Module):
    """The complete net we use for predicting the disparity of a stereo input.

    Input: two tensors of the shape (B, 3, H, W)

    Output: disparity map of shape (B, H, W)"""

    def __init__(self, use_resnet=False,
                 fix_resnet=False,
                 channel_fe=[3,4,4,8,8,16,16,16,32],
                 channel_cp=[64, 32, 32, 32, 1],
                 dropout_p=0.5):
        """
        :param channel_fe: The channel sizes of the feature extractor
        :param kernel_fe: The kernel sizes of the feature extractor
        :param pool_layers: The pooling layers of the feature extractor
        :param pool_type: The pooling type of the feature extractor uses
                          (either 'avg' or 'max')
        :param channel_cp: The channel sizes of the cost processing
        :param kernel_cp: The kernel sizes of the cost processing
        """
        super().__init__()
        if use_resnet:
            self.feature_extraction = ResnetFE(fix_resnet)
        else:
            self.feature_extraction = FeatureExtractor(channels=channel_fe,
                                                       dropout_p=dropout_p)
        self.cost_processing = CostProcessing(channels=channel_cp,
                                              dropout_p=dropout_p)
        self.regressor = Regressor()

    def forward(self, left, right):
        left_feats = self.feature_extraction(left)
        right_feats = self.feature_extraction(right)

        cost = self.cost_processing(left_feats, right_feats)
        disp = self.regressor(cost)
        return disp
