# inspired by https://github.com/JiaRenChang/PSMNet/blob/master/dataloader/SecenFlowLoader.py
# though this is somewhat changed
import torch
import torch.utils.data as data
import torch
from PIL import Image
from . import preprocess
from . import readpfm as rp
import numpy as np
import numpy.random as rand


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return rp.readPFM(path)

class myImageFolder(data.Dataset):

    def __init__(self, left, right, left_disparity, training, right_disparity=None, loader=default_loader, dploader= disparity_loader):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.disp_R = right_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        if training:
            self.preprocess = preprocess.Transformation(augment=True, center_crop=False)
        else:
            self.preprocess = preprocess.Transformation(augment=False, center_crop=False)

    def __getitem__(self, index):

        left  = self.left[index]
        right = self.right[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        if self.disp_R is not None and rand.randint(2)==0:
            # with a probability of 0.5, we flip and swap the images and
            # use the flipped right disparity
            left_img, right_img = right_img.transpose(Image.FLIP_LEFT_RIGHT), left_img.transpose(Image.FLIP_LEFT_RIGHT)

            disp_L= self.disp_R[index] #has to be flipped
            dataL, scaleL = self.dploader(disp_L)
            dataL = np.fliplr(dataL)
        else:
            disp_L= self.disp_L[index]
            dataL, scaleL = self.dploader(disp_L)

        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        left_img, right_img, dataL = self.preprocess(left_img, right_img, dataL)

        return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
