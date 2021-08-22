import torch
from PIL import Image
import os.path
from . import preprocess
import sys
sys.path.insert(0, '../')
from my_utils import device, base_path

left_folder = 'datasets/kitti2015/training/image_2/'
right_folder = 'datasets/kitti2015/training/image_3/'
disp_folder = 'datasets/kitti2015/training/disp_occ_0/'

class KittiDataset(torch.utils.data.Dataset):
    """A class loading the KITTI 2015 scene flow dataset.
    """

    def __init__(self, set_type):
        """
        :param set_type: Indicates whether the dataset is used for training,
                         validation or testing (can be set to 'train', 'val' or 'test').
        """
        assert set_type in ['train', 'val', 'test']
        self.set_type = set_type

        if set_type == "test":
            img_indices = list(range(150, 200))
            self.preprocess = preprocess.Transformation(augment=False,
                                                        center_crop=True)
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
                self.preprocess = preprocess.Transformation(augment=True,
                                                            center_crop=False)
            if set_type == 'val':
                img_indices = possible_indices[125:]
                self.preprocess = preprocess.Transformation(augment=False,
                                                            center_crop=True)

        self.length = len(img_indices)

        self.left_images = []
        self.right_images = []
        self.left_disparity = []

        for file_num in img_indices:
            file_name = str(file_num).rjust(6, '0') + '_10.png'

            self.left_images.append(os.path.join(base_path, left_folder, file_name))
            self.right_images.append(os.path.join(base_path, right_folder, file_name))
            self.left_disparity.append(os.path.join(base_path, disp_folder, file_name))

    def __getitem__(self, idx):
        left = Image.open(self.left_images[idx])
        right = Image.open(self.right_images[idx])
        disp = Image.open(self.left_disparity[idx])

        left, right, disp = self.preprocess(left, right, disp)
        return left, right, disp/256.0

    def __len__(self):
        return self.length
