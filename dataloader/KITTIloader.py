import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm
import os.path
from .my_utils import device
import numpy as np

left_folder = 'datasets/kitti2015/training/image_2/'
right_folder = 'datasets/kitti2015/training/image_3/'
disp_folder = 'datasets/kitti2015/training/disp_occ_0/'

class KittiDataset(torch.utils.data.Dataset):
    """A class loading the KITTI 2015 scene flow dataset.
    """

    def __init__(self, set_type, transform=None):
        """
        We advise to use the following transformation for training:
            transform = transforms.Compose([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            )
        We advise to use the following transformation for testing and validation:
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

        Other than normalizations and color-jitters, no transformations are allowed.
        The images automatically get reduced to size (256, 512) in training and
        to size (376, 1240) in testing and validation.
        """
        assert set_type in ['train', 'val', 'test']
        self.set_type = set_type
        self.transform = transform

        if set_type == "test":
            img_indices = list(range(150, 200))
        else:
            # a random permutation of range(150) to split the data randomly into
            # validation and training sets
            # (we hardcoded this in order to ensure comparability):
            possible_indices = [54, 128, 146, 79, 28, 71, 18, 52, 62, 76,
                                99, 116, 26, 127, 97, 90, 143, 25, 13, 9,
                                105, 134, 16, 107, 39, 135, 58, 140, 12, 147,
                                136, 30, 74, 109, 106, 123, 122, 130, 72, 65,
                                87, 22, 121, 6, 15, 56, 139, 119, 31, 53,
                                29, 144, 27, 77, 81, 51, 73, 63, 68, 32,
                                23, 59, 118, 37, 131, 21, 129, 24, 89, 11,
                                43, 84, 82, 5, 49, 85, 101, 60, 145, 17,
                                120, 10, 113, 132, 93, 100, 14, 112, 149, 125,
                                102, 57, 117, 38, 4, 96, 91, 95, 104, 36,
                                67, 50, 48, 78, 148, 94, 137, 115, 75, 108,
                                0, 83, 141, 1, 92, 114, 110, 124, 98, 111,
                                20, 47, 142, 8, 86, 133, 66, 46, 7, 69,
                                44, 33, 2, 40, 61, 3, 55, 80, 138, 19,
                                42, 126, 35, 103, 88, 64, 41, 70, 34, 45]
            if set_type == 'train':
                img_indices = possible_indices[:125]
            if set_type == 'val':
                img_indices = possible_indices[125:]

        self.length = len(img_indices)

        self.left_images = []
        self.right_images = []
        self.left_disparity = []



        for file_num in img_indices:
            file_name = str(file_num).rjust(6, '0') + '_10.png'

            self.left_images.append(os.path.join(left_folder, file_name))
            self.right_images.append(os.path.join(right_folder, file_name))
            self.left_disparity.append(os.path.join(disp_folder, file_name))

        if self.set_type in ['test', 'val']:
            self.center_crop = torchvision.transforms.CenterCrop((376, 1240))

    def __getitem__(self, idx):
        left = to_tensor(Image.open(self.left_images[idx]))
        right = to_tensor(Image.open(self.right_images[idx]))
        disp = torch.squeeze(to_tensor(Image.open(self.left_disparity[idx])))

        if self.set_type in ['test', 'val']:
            left_right_image = torch.empty([2, 3, 376, 1240])
            # disparity = self.center_crop(self.left_disparity[idx])
            disparity = self.center_crop(disp)

            # left_right_image[0] = self.center_crop(self.left_images[idx])
            left_right_image[0] = self.center_crop(left)
            # left_right_image[1] = self.center_crop(self.right_images[idx])
            left_right_image[1] = self.center_crop(right)
        else:
            # the set type is training
            # here we construct a random patch of size (256, 512) of the image

            # orig_shape = self.left_images[0].shape
            orig_shape = left.shape
            start_h = np.random.randint(orig_shape[-2] - 256)
            end_h = start_h + 256
            start_w = np.random.randint(orig_shape[-1] - 512)
            end_w = start_w + 512

            #left_image = self.left_images[idx][:, start_h:end_h, start_w:end_w]
            #right_image = self.right_images[idx][:, start_h:end_h, start_w:end_w]
            #disparity = self.left_disparity[idx][start_h:end_h, start_w:end_w]

            left_image = left[:, start_h:end_h, start_w:end_w]
            right_image = right[:, start_h:end_h, start_w:end_w]
            disparity = disp[start_h:end_h, start_w:end_w]

            left_right_image = torch.empty([2, 3, 256, 512])
            left_right_image[0] = left_image
            left_right_image[1] = right_image

        if self.transform is not None:
            left_right_image = self.transform(left_right_image)

        return left_right_image[0], left_right_image[1], disparity

    def __len__(self):
        return self.length
