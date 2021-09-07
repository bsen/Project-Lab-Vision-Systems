# inspired by https://github.com/JiaRenChang/PSMNet/blob/master/dataloader/preprocess.pytrast
# but heavily changed
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import numpy.random as rand
import torchvision.transforms.functional as F
from torchvision.transforms.functional import to_tensor
from PIL import Image


imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class Transformation():

    global imagenet_stats

    def __init__ (self, rand_crop=True, col_jitter_flip=False, center_crop=False):
        assert not (rand_crop and center_crop) # random and center_crop can
                                               # not be used together
            
        self.rand_crop = rand_crop
        self.col_jitter_flip = col_jitter_flip
        self.center_crop = center_crop
        if rand_crop:
            self.random_crop = RandomCrop(256, 512)
        if col_jitter_flip:
            self.color_jitter = ColorJitter(0.2, 0.2, 0.2, 0.05)

        if center_crop:
            self.normalize = transforms.Normalize(**imagenet_stats)
            self.center_crop = CenterCrop(376, 1240)
        else:
            self.normalize = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(**imagenet_stats)])

    def __call__(self, left, right, disp):
        if self.center_crop:
            left, right, disp = self.center_crop(left, right, disp)

        if not self.center_crop:
            disp = torch.squeeze(to_tensor(disp))
        if self.rand_crop:
            left, right, disp = self.random_crop(left, right, disp)
        if self.col_jitter_flip:
            left, right = self.color_jitter(left, right)
            left, right, disp = random_flip(left, right, disp)

        return self.normalize(left), self.normalize(right), torch.squeeze(disp)

class CenterCrop():
    def __init__(self, th, tw):
        self.crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop([th, tw])
            ])
    def __call__(self, left, right, disp):
        disp = self.crop(disp)
        left = self.crop(left)
        right = self.crop(right)
        return left, right, disp


class RandomCrop():
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, left, right, disp):
        w, h = left.size

        w_low = random.randint(0, w - self.tw)
        h_low = random.randint(0, h - self.th)

        w_top = w_low + self.tw
        h_top = h_low + self.th

        left = left.crop((w_low, h_low, w_top, h_top))
        right = right.crop((w_low, h_low, w_top, h_top))

        disp = disp[h_low:h_top, w_low:w_top]
        return left, right, disp


class ColorJitter(object):
    def __init__(self, brightness, contrast, saturation, hue):
        self.b_max = 1+brightness
        self.b_min = max(0.0, 1-brightness)
        self.c_max = 1+contrast
        self.c_min = max(0.0, 1-contrast)
        self.s_max = 1+saturation
        self.s_min = max(0.0, 1-saturation)
        self.h_max = hue
        self.h_min = hue

    def __call__(self, left, right):
        b = rand.uniform(self.b_min, self.b_max)
        c = rand.uniform(self.c_min, self.c_max)
        s = rand.uniform(self.s_min, self.s_max)
        h = rand.uniform(self.h_min, self.h_max)

        left = F.adjust_brightness(left, b)
        left = F.adjust_contrast(left, c)
        left = F.adjust_saturation(left, s)
        left = F.adjust_hue(left, h)

        right = F.adjust_brightness(right, b)
        right = F.adjust_contrast(right, c)
        right = F.adjust_saturation(right, s)
        right = F.adjust_hue(right, h)
        return left, right

def random_flip(left, right, disparity):
    """
    With probability 0.5 returns left, right and disparity flipped on upside down.
    """
    if rand.randint(2)==0:
        left = left.transpose(Image.FLIP_TOP_BOTTOM)
        right = right.transpose(Image.FLIP_TOP_BOTTOM)
        disparity = torch.flip(disparity, dims=[-2])
    return left, right, disparity