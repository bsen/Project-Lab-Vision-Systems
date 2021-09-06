import torch
import torch.nn as nn
import torch.nn.functional as F


class Regressor(nn.Module):
    """
    The disparity regression module.
    Input is a tensor of shape (B, 1, Disp/4, H/4, W/4).

    We first upsample to size (B, Disp, H, W).
    Then we do a channelwise softmax on this.
    Then we weight and sum over the channels to get a output of size (B, 1, H, W).

    Output is the disparity map of size (B, H, W)
    """

    def __init__(self, dropout_p):
        super().__init__()
        
        self.upsample = nn.Sequential(
                            nn.ConvTranspose2d(48, 96, 3, stride=2, padding=1, output_padding=1),
                            nn.BatchNorm2d(96),
                            nn.Dropout(dropout_p),
                            nn.ReLU(),
                            nn.Conv2d(96, 96, 3, padding=1),
                            nn.BatchNorm2d(96),
                            nn.Dropout(dropout_p),
                            nn.ReLU(),
                            nn.ConvTranspose2d(96, 192, 3, stride=2, padding=1, output_padding=1)
        )
        self.weight_sum = nn.Conv2d(in_channels=192, out_channels=1, kernel_size=1, bias=False)
        
        
        
    def forward(self, x):
        B, _, _, H4, W4 = x.shape
        Disp = 192
        H = H4*4
        W = W4*4

        # upsample
        #cost = F.interpolate(x, size=[Disp, H, W], mode='trilinear')
        #cost = torch.squeeze(cost, dim=1)
        cost = torch.squeeze(x, dim=1)
        cost = self.upsample(cost)
        

        # channelwise softmax
        cost = F.softmax(cost, dim=1)

        # weight & sum
        output = self.weight_sum(cost)
        
        return torch.squeeze(output, dim=1)
