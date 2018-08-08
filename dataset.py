#-*- coding: UTF-8 -*-
import os
from imageio import imread, imwrite
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from skimage import img_as_ubyte
from config import opt

class My_Dataset(data.Dataset):
    def __init__(self, img_path, mask_path, transforms = None):
        super(My_Dataset, self).__init__()
        image_list = [img for img in os.listdir(img_path)]
        self.big_imgs = [os.path.join(img_path, img) for img in image_list]
        self.big_masks = [os.path.join(mask_path, '{}{}'.format(img.split('.')[0], '.tif')) for img in image_list]
        self.r_size = opt.img_rsize
        self.c_size = opt.img_csize
        self.jiange = opt.crop_jiange
        self.image_dict, self.mask_dict = self.get_init_dict()
        self.data_index = self.get_indexs()

        self.transforms = T.Compose([T.ToPILImage(),T.ToTensor()])
        print("length:{}".format(len(self.data_index)))

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, item):
        index = self.data_index[item]
        crop_img = self.get_crop(self.image_dict[index[0]], index[1], index[2])
        crop_mask = self.get_crop(self.mask_dict[index[0]], index[1], index[2])
        crop_mask = np.expand_dims(crop_mask, axis=2)
        data = self.transforms(crop_img)
        label = self.transforms(crop_mask)
        return data, label

    def get_init_dict(self):
        image_dict = dict().fromkeys(list(range(0, len(self.big_imgs))), None)
        mask_dict = dict().fromkeys(list(range(0, len(self.big_imgs))), None)
        for i in range(0, len(self.big_imgs)):
            image_dict[i] = imread(self.big_imgs[i])
            mask_dict[i] = img_as_ubyte(imread(self.big_masks[i])/255)
        return image_dict, mask_dict

    def get_indexs(self):
        r, c, _ = imread(self.big_imgs[0]).shape
        r_number = (r-self.r_size)//self.jiange + 1
        c_number = (c-self.c_size)//self.jiange + 1
        indexs = [(i, j, k) for i in range(0, len(self.big_imgs))
                            for j in range(0, r_number)
                            for k in range(0, c_number)]
        return indexs

    def get_crop(self, img, r_num, c_num):
        r_index = r_num  * self.jiange
        c_index = c_num  * self.jiange
        crop_img = img[r_index:r_index + self.r_size, c_index:c_index + self.c_size]
        return crop_img
