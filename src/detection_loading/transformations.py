# Copyright (C) 2020 Yanqi Xu, Yiqiu Shen, Laura Heacock, Carlos Fernandez-Granda, Krzysztof J. Geras

# This file is part of Mammo-DETR.
#
# Mammo-DETR is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Mammo-DETR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Mammo-DETR.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as F
import random

import torch

class Standardizer(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        img = sample["img"]
        img = (img - img.mean()) / np.maximum(img.std(), 10 ** (-5))
        sample["img"] = img
        return sample


class CopyChannel(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        if sample["img"].size()[0] == 1:
            sample["img"] = sample["img"].repeat([3, 1, 1])
        return sample


class ToNumpy(object):
    """
    Use this class to shut up "UserWarning: The given NumPy array is not writeable ..."
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        sample["img"] = np.array(sample["img"])
        sample["bseg"] = np.array(sample["bseg"])
        sample["mseg"] = np.array(sample["mseg"])
        return sample


class Resize:
    def __init__(self, size):
        self.resize_func = transforms.Resize(size)

    def __call__(self, sample):
        sample["img"] = self.resize_func(sample["img"])
        sample["bseg"] = self.resize_func(sample["bseg"])
        sample["mseg"] = self.resize_func(sample["mseg"])
        return sample


class ToTensor:
    def __init__(self):
        self.operator = transforms.ToTensor()

    def __call__(self, sample):
        sample["img"] = self.operator(sample["img"])
        sample["bseg"] = self.operator(sample["bseg"])
        sample["mseg"] = self.operator(sample["mseg"])
        return sample


class RandomAffine(transforms.RandomAffine):
    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, sample["img"].size)
        sample["img"] = F.affine(sample["img"], *ret)
        sample["bseg"] = F.affine(sample["bseg"], *ret)
        sample["mseg"] = F.affine(sample["mseg"], *ret)
        return sample


class RandomFlip:
    def __init__(self, hprob=0.5, vprob=0.5):
        self.hprob = hprob
        self.vprob = vprob

    def __call__(self, sample):
        hrandom = np.random.rand()
        vrandom = np.random.rand()
        if hrandom < self.hprob :
            sample["img"] = F.hflip(sample["img"])
            sample["mseg"] = F.hflip(sample["mseg"])
            sample["bseg"] = F.hflip(sample["bseg"])
        if vrandom < self.vprob :
            sample["img"] = F.vflip(sample["img"])
            sample["mseg"] = F.vflip(sample["mseg"])
            sample["bseg"] = F.vflip(sample["bseg"])
        return sample


class RandomGrayScale:
    def __init__(self, p=0.2, num_output_channels=3, generator=None):
        self.p = p
        if generator is None:
            self.generator = np.random.default_rng(seed=666)
        else:
            self.generator = generator
        self.transform = transforms.Grayscale(num_output_channels=num_output_channels)

    def __call__(self, sample):
        if self.generator.random() < self.p:
            sample["img"] = self.transform(sample["img"])
        return sample


class RandomGaussianBlur:
    def __init__(self, p=0.5, generator=None):
        self.p = p
        if generator is None:
            self.generator = np.random.default_rng(seed=666)
        else:
            self.generator = generator
        self.transform = transforms.GaussianBlur(sigma=(0.1, 2.0), kernel_size=(23,23))

    def __call__(self, sample):
        if np.random.rand() < self.p:
            sample["img"] = self.transform(sample["img"])
        return sample


class RandomErasing:
    def __init__(self, p, scale, ratio, generator=None):
        """
        https://pytorch.org/vision/main/generated/torchvision.transforms.RandomErasing.html
        :param p:
        :param scale: range of proportion of erased area against input image.
        :param ratio: range of aspect ratio of erased area.
        :param generator:
        """
        self.p = p
        self.scale = scale
        self.ratio = ratio
        if generator is None:
            self.generator = np.random.default_rng(seed=666)
        else:
            self.generator = generator

    def __call__(self, sample):
        if self.generator.random() < self.p:
            region = transforms.RandomErasing.get_params(sample["img"], scale=self.scale, ratio=self.ratio)
            i, j, h, w, v = region
            # mask out image[:,  i->i+h, j->j+w]
            F.erase(sample["img"], i, j, h, w, v, inplace=True)
            sample["bseg"][0, i:i+h, j:j+w] = 0
            sample["mseg"][0, i:i+h, j:j+w] = 0
        return sample


class RandomResizedCrop:
    def __init__(self, p, scale, ratio, generator=None):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        if generator is None:
            self.generator = np.random.default_rng(seed=666)
        else:
            self.generator = generator

    def __call__(self, sample):
        if self.generator.random() < self.p:
            i, j, h, w = transforms.RandomResizedCrop.get_params(sample["img"], self.scale, self.ratio)
      
            h_img, w_img = sample["img"].size

            #_, h_img, w_img = sample["img"].size()
            
            sample["img"] = F.crop(sample["img"], i, j, h, w)
            sample["img"] = F.resize(sample["img"], (h_img, w_img))
            sample["mseg"] = F.crop(sample["mseg"], i, j, h, w)
            sample["mseg"] = F.resize(sample["mseg"], (h_img, w_img))
            sample["bseg"] = F.crop(sample["bseg"], i, j, h, w)
            sample["bseg"] = F.resize(sample["bseg"], (h_img, w_img))
        return sample


omni_augmentation = [RandomFlip(0.5, 0),
                     #RandomGrayScale(p=0.2), # this blacks out the entire image
                     #RandomGaussianBlur(p=0.5) # this is very slow
                     RandomResizedCrop(p=0.5, scale=(0.5, 1.0), ratio=(0.75, 1.333))
                     ]
omni_tensor_transformation = [
                     RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3)),
                     RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6)),
                     RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8))
                              ]
                      
standard_augmentation = [
    RandomFlip(0.5, 0.5),
    RandomAffine(degrees=(-45, 45), translate=(0.1,0.1), scale=(0.7, 1.5), shear=(-25, 25)),
]

medium_augmentation = [
    RandomFlip(0.5, 0),
    RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.8, 1.3), shear=(-10, 10)),
]

weak_augmentation = [
    RandomFlip(0.5, 0.5),
    RandomAffine(degrees=(-10, 10), translate=(0.1,0.1), scale=(0.9, 1.1)),
]

test_time_augmentation = [
    RandomFlip(0.5, 0.5), RandomAffine(degrees=(-15, 15)),
    ]

to_tensor = [ToNumpy(), ToTensor(), Standardizer()]


def compose_transform(augmentation=None, resize=None, image_format="greyscale", copy_paste=False):
    basic_transform = []
    # add augmentation
    if augmentation is not None:
        if augmentation == "standard":
            basic_transform += standard_augmentation
        elif augmentation == "omni":
            basic_transform += omni_augmentation
        elif augmentation == "mid":
            basic_transform += medium_augmentation
        elif augmentation == "weak":
            basic_transform += weak_augmentation
        elif augmentation == "test_time":
            basic_transform += test_time_augmentation
        else:
            raise ValueError("invalid augmentation {}".format(augmentation))

    # add resize
    if resize is not None:
        basic_transform += [Resize(resize)]

 
    # add to tensor and normalization
    basic_transform += to_tensor

    # tensor processing
    if copy_paste:
        basic_transform += [CopyPaste()]

    if augmentation is not None:
        if augmentation == "omni":
            basic_transform += omni_tensor_transformation
   

    # add channel duplication
    if image_format == "greyscale":
        basic_transform += [CopyChannel()]
    return transforms.Compose(basic_transform)

