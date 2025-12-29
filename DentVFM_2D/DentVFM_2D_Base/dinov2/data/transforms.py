# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
from torchvision import transforms
import numpy as np
import random
from PIL import Image

class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# 20250909自定义transform类
class PatchCovering(object):
    def __init__(self, num_patches=20):
        self.num_patches = num_patches

    def generate_gray_patch(self, image, min_size, max_size):
        """
        随机生成一个灰色补丁，补丁的大小在指定的最小和最大尺寸之间
        """
        # 随机生成补丁的宽度和高度
        patch_width = random.randint(int(min_size[0]), int(max_size[0]))
        patch_height = random.randint(int(min_size[1]), int(max_size[1]))
        
        # 生成一个灰色补丁，灰度值为128（中灰色）
        patch = np.full((patch_height, patch_width, 3), 128, dtype=np.uint8)
        
        # 随机选择补丁的位置
        max_x = image.shape[1] - patch_width
        max_y = image.shape[0] - patch_height
        x_pos = random.randint(0, max_x)
        y_pos = random.randint(0, max_y)
        
        # 将补丁覆盖到图像上
        image[y_pos:y_pos + patch_height, x_pos:x_pos + patch_width] = patch
        
        return image

    def __call__(self, image):
        """
        对图像进行patch covering增强，覆盖病变区域和健康区域
        :param image: 输入图像（PIL图像）
        :return: 增强后的图像（PIL图像）
        """
        image = np.array(image)  # 转换为 NumPy 数组
        
        # 获取病变区域和健康区域的尺寸
        width = image.shape[1]
        height = image.shape[0]
        
        # 生成补丁覆盖病变区域和健康区域
        for _ in range(self.num_patches // 2):
            # 随机覆盖病变区域
            image = self.generate_gray_patch(image, (width*0.045, height*0.085), (width*0.055, height*0.095))
            # 随机覆盖健康区域
            image = self.generate_gray_patch(image, (width*0.045, height*0.085), (width*0.055, height*0.095))
        
        return Image.fromarray(image)  # 转回为 PIL 图像 

    

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    # 原始
    # transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    
    # 250905用于改善cyst
    transforms_list = [transforms.RandomResizedCrop(crop_size, scale=(0.6, 1.0), ratio=(0.9, 1.1), interpolation=interpolation, antialias=True)]
    # transforms_list = [PatchCovering(), transforms.RandomResizedCrop(crop_size, scale=(0.6, 1.0), ratio=(0.9, 1.1), interpolation=interpolation, antialias=True)]
    
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    
    transforms_list.extend(
        [
            # 250905cyst添加
            # transforms.RandomRotation(10, interpolation=interpolation),
            # transforms.RandomAutocontrast(p=0.2),
            # transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.2),
            
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)
