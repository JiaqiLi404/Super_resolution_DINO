# @Time : 2023/4/22 19:22 
# @Author : Li Jiaqi
# @Description :
from torch.utils.data import Dataset
import glob
import os
from skimage.transform import resize
from skimage.io import imread
from torch.utils.data import DataLoader
import torch
from collections import OrderedDict
import numpy as np
import logging

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)

LABEL_MAP = OrderedDict(
    Background=0,
    Building=1,
    Road=2,
    Water=3,
    Barren=4,
    Forest=5,
    Agricultural=6
)


class SitesBingBook(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.id_list = []
        png_list = glob.glob(os.path.join(data_dir, '*.png'))
        self.id_list = [os.path.split(fp)[-1][:-8] for fp in png_list]

        self.transforms = transforms

    def __getitem__(self, idx):
        file_id = self.id_list[idx]
        bing_file_name = file_id + 'bing.png'
        book_file_name = file_id + 'book.jpg'
        bing_image = imread(os.path.join(self.data_dir, bing_file_name))
        bing_image = bing_image[:-23, :, 0:3]  # delete the alpha dimension in png files and bing flag
        book_image = imread(os.path.join(self.data_dir, book_file_name))
        book_image = book_image[:-75, :]  # delete the book flag

        # 2d black and white book images to 3d images
        if len(book_image.shape) <= 2:
            new_book_image = np.zeros(dtype=np.uint8, shape=(book_image.shape[0], book_image.shape[1], 3))
            new_book_image[:, :, 0] = book_image * 255
            new_book_image[:, :, 1] = book_image * 255
            new_book_image[:, :, 2] = book_image * 255
            book_image = new_book_image
        elif book_image.dtype != np.uint8:
            book_image = book_image * 255

        # normalize and resize
        if self.transforms is not None:
            blob = self.transforms(image=bing_image)
            bing_image = blob['image']
            blob = self.transforms(image=book_image)
            book_image = blob['image']

        # to C,W,H
        bing_image = np.rollaxis(bing_image, 2, 0)
        book_image = np.rollaxis(book_image, 2, 0)

        return bing_image, book_image

    def __len__(self):
        return len(self.id_list)


class SitesLoader(DataLoader):
    def __init__(self, config, flag="train"):
        self.config = config
        self.flag = flag
        dataset = SitesBingBook(self.config["dataset"], self.config["transforms"])
        super(SitesLoader, self).__init__(dataset,
                                          batch_size=self.config['batch_size'],
                                          num_workers=self.config['num_workers'],
                                          shuffle=self.config['shuffle'],
                                          pin_memory=self.config['pin_memory'],
                                          drop_last=self.config['drop_last']
                                          )
