# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

## 参考dinov2 3d的数据增强

import logging

from torchvision import transforms
from monai.transforms import (
    Randomizable,
    Crop,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    SpatialPadd,
    ToTensord,
    RandFlipd,
    OneOf,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandHistogramShiftd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandGibbsNoised,
    Lambdad
)
from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)
from monai.data.utils import get_random_patch, get_valid_patch_size
from torch.nn.functional import interpolate
import math
import torch

logger = logging.getLogger("dinov2")


class RandomResizedCrop3d(Crop, Randomizable):
    def __init__(
        self,
        size,
        in_slice_scale,
        cross_slice_scale,
        interpolation='trilinear',
        aspect_ratio=(0.9, 1/0.9),
    ):
        """
        Adapting torch RandomResizedCrop to 3D data by separating in-slice/in-plane and cross-slice dimensions.

        Args:
            size: Size of output image.
            in_slice_scale: Range of the random size of the cropped in-slice/in-plane dimensions.
            cross_slice_scale: Range of the random size of the cropped cross-slice dimensions.
            interpolation: 3D interpolation method, defaults to 'trilinear'.
            aspect_ratio: Range of aspect ratios of the cropped in-slice/in-plane dimensions.
        """
        super().__init__()
        self.size = size
        self.in_slice_scale = in_slice_scale
        self.cross_slice_scale = cross_slice_scale
        self.interpolation = interpolation
        self.aspect_ratio = aspect_ratio
        self._slices: tuple[slice, ...] = ()

    def get_in_slice_crop(self, height, width):
        """
        Adapted from torchvision RandomResizedCrop, applied to the in-slice/in-plane dimensions
        """
        area = height * width

        log_ratio = math.log(self.aspect_ratio[0]), math.log(self.aspect_ratio[1])
        for _ in range(10):
            target_area = area * self.R.uniform(*self.in_slice_scale)
            aspect_ratio = math.exp(self.R.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                return h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.aspect_ratio):
            w = width
            h = int(round(w / min(self.aspect_ratio)))
        elif in_ratio > max(self.aspect_ratio):
            h = height
            w = int(round(h * max(self.aspect_ratio)))
        else:  # whole image
            w = width
            h = height
        return h, w

    def randomize(self, img_size):
        # first two dimensions are dicom slice dims/in-plane dims, third is number of slices
        height, width, depth = img_size

        # get in-slice crop size
        crop_h, crop_w = self.get_in_slice_crop(height, width)

        # get cross-slice crop size
        crop_d = int(round(depth * self.R.uniform(*self.cross_slice_scale)))

        crop_size = (crop_h, crop_w, crop_d)
        valid_size = get_valid_patch_size(img_size, crop_size)
        self._slices = get_random_patch(img_size, valid_size, self.R)

    def __call__(self, img, lazy=False):
        self.randomize(img.shape[1:])
        cropped = super().__call__(img=img, slices=self._slices)
        resized = interpolate(cropped.unsqueeze(0), size=self.size, mode=self.interpolation).squeeze(0)
        return resized


# 数据transform 旧版，不涉及cache
class DataAugmentationDINO3D_old(object):
    def __init__(
            self,
            preprocessing,
            local_crops_number,
            global_crops_size=224,
            local_crops_size=96,
        ):
        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")
        self.preprocessing = preprocessing
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.local_crops_number = local_crops_number
        
        # 一些transform操作：加载图像-通道变换-方向变换-重采样-强度归一化-边界框裁剪
        self.init_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),

                # 新增去除nan
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),

                Spacingd(keys=["image"], pixdim=(self.preprocessing.space_x, self.preprocessing.space_y, self.preprocessing.space_z), mode=("bilinear")),
                
                # 新增clip
                # ScaleIntensityRanged(
                #     keys=["image"], a_min=self.preprocessing.a_min, a_max=self.preprocessing.a_max, b_min=self.preprocessing.b_min, b_max=self.preprocessing.b_max, clip=True
                # ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=self.preprocessing.b_min, b_max=self.preprocessing.b_max, clip=True),

                CropForegroundd(keys=["image"], source_key="image"),
            ]
        )

        # transform操作：过小的图像进行pad-随机crop-图像旋转90度-图像强度偏移
        self.global_transforms1 = Compose(
            [
                SpatialPadd(keys="image", spatial_size=[self.global_crops_size, self.global_crops_size, self.global_crops_size]),
                RandSpatialCropSamplesd(
                    keys=["image"],
                    roi_size=[self.global_crops_size, self.global_crops_size, self.global_crops_size],
                    num_samples=1,
                    random_center=True,
                    random_size=False,
                ),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[0]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[1]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[2]),
                # RandRotate90d(keys=["image"], prob=0.10, max_k=3,),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(1, 2)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 2)),
                OneOf(
                        [
                            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                            RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                            RandHistogramShiftd(keys=["image"], num_control_points=10, prob=0.3),
                        ]
                    ),
                OneOf(
                        [
                            RandGaussianSmoothd(keys=["image"],prob=0.3),
                            RandGaussianSharpend(keys=["image"],prob=0.3),
                        ]
                    )
            ]
        )
        self.global_transforms2 = Compose(
            [
                SpatialPadd(keys="image", spatial_size=[self.global_crops_size, self.global_crops_size, self.global_crops_size]),
                RandSpatialCropSamplesd(
                    keys=["image"],
                    roi_size=[self.global_crops_size, self.global_crops_size, self.global_crops_size],
                    num_samples=1,
                    random_center=True,
                    random_size=False,
                ),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[0]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[1]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[2]),
                # RandRotate90d(keys=["image"], prob=0.10, max_k=3,),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(1, 2)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 2)),
                OneOf(
                        [
                            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                            RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                            RandHistogramShiftd(keys=["image"], num_control_points=10, prob=0.3),
                        ]
                    ),
                OneOf(
                        [
                            RandGaussianSmoothd(keys=["image"],prob=0.3),
                            RandGaussianSharpend(keys=["image"],prob=0.3),
                        ]
                    ),
                RandGibbsNoised(keys=["image"], prob=0.2)
            ]
        )
        
        # 局部裁剪transformer：pad-crop-旋转-强度偏移
        self.local_transforms = Compose(
            [
                SpatialPadd(keys="image", spatial_size=[self.local_crops_size, self.local_crops_size, self.local_crops_size]),
                RandSpatialCropSamplesd(
                    keys=["image"],
                    roi_size=[self.local_crops_size, self.local_crops_size, self.local_crops_size],
                    num_samples=self.local_crops_number,
                    random_center=True,
                    random_size=False,
                ),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[0]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[1]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[2]),
                # RandRotate90d(keys=["image"], prob=0.10, max_k=3,),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(1, 2)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 2)),
                OneOf(
                        [
                            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                            RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                            RandHistogramShiftd(keys=["image"], num_control_points=10, prob=0.3),
                        ]
                    ),
                RandGaussianSmoothd(keys=["image"], prob=0.3)
            ]
        )

        self.to_tensor = Compose(
            [
                ToTensord(keys=["image"])
            ]
        )
        
        self.global_transfo1 = Compose([self.global_transforms1, self.to_tensor])
        self.global_transfo2 = Compose([self.global_transforms2, self.to_tensor])
        self.local_transfo = Compose([self.local_transforms, self.to_tensor])

    def __call__(self, data):
        output = {}
        
        # global crops:
        im_base = self.init_transforms(data)
        global_crop_1 = self.global_transfo1(im_base)[0]["image"]
        global_crop_2 = self.global_transfo2(im_base)[0]["image"]

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]
        local_crops = [ each["image"]
             for each in self.local_transfo(im_base)
        ]
        # local crops:
        # local_crops = [
        #     self.local_transfo(im_base) for _ in range(self.local_crops_number)
        # ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output        

## 新版涉及cache
class DataAugmentationDINO3D(object):
    def __init__(
            self,
            preprocessing,
            local_crops_number,
            global_crops_size=224,
            local_crops_size=96,
        ):
        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")
        self.preprocessing = preprocessing
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.local_crops_number = local_crops_number
        
        # 一些transform操作：加载图像-通道变换-方向变换-重采样-强度归一化-边界框裁剪
        # self.init_transforms = Compose(
        #     [
        #         LoadImaged(keys=["image"]),
        #         EnsureChannelFirstd(keys=["image"]),
        #         Orientationd(keys=["image"], axcodes="RAS"),

        #         # 新增去除nan
        #         Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),

        #         Spacingd(keys=["image"], pixdim=(self.preprocessing.space_x, self.preprocessing.space_y, self.preprocessing.space_z), mode=("bilinear")),
                
        #         # 新增clip
        #         # ScaleIntensityRanged(
        #         #     keys=["image"], a_min=self.preprocessing.a_min, a_max=self.preprocessing.a_max, b_min=self.preprocessing.b_min, b_max=self.preprocessing.b_max, clip=True
        #         # ),
        #         ScaleIntensityRangePercentilesd(
        #             keys=["image"], lower=0.05, upper=99.95, b_min=self.preprocessing.b_min, b_max=self.preprocessing.b_max, clip=True),

        #         CropForegroundd(keys=["image"], source_key="image"),
        #     ]
        # )

        # transform操作：过小的图像进行pad-随机crop-图像旋转90度-图像强度偏移
        self.global_transforms1 = Compose(
            [
                SpatialPadd(keys="image", spatial_size=[self.global_crops_size, self.global_crops_size, self.global_crops_size]),
                RandSpatialCropSamplesd(
                    keys=["image"],
                    roi_size=[self.global_crops_size, self.global_crops_size, self.global_crops_size],
                    num_samples=1,
                    random_center=True,
                    random_size=False,
                ),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[0]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[1]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[2]),
                # RandRotate90d(keys=["image"], prob=0.10, max_k=3,),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(1, 2)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 2)),
                OneOf(
                        [
                            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                            RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                            RandHistogramShiftd(keys=["image"], num_control_points=10, prob=0.3),
                        ]
                    ),
                OneOf(
                        [
                            RandGaussianSmoothd(keys=["image"],prob=0.3),
                            RandGaussianSharpend(keys=["image"],prob=0.3),
                        ]
                    )
            ]
        )
        self.global_transforms2 = Compose(
            [
                SpatialPadd(keys="image", spatial_size=[self.global_crops_size, self.global_crops_size, self.global_crops_size]),
                RandSpatialCropSamplesd(
                    keys=["image"],
                    roi_size=[self.global_crops_size, self.global_crops_size, self.global_crops_size],
                    num_samples=1,
                    random_center=True,
                    random_size=False,
                ),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[0]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[1]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[2]),
                # RandRotate90d(keys=["image"], prob=0.10, max_k=3,),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(1, 2)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 2)),
                OneOf(
                        [
                            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                            RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                            RandHistogramShiftd(keys=["image"], num_control_points=10, prob=0.3),
                        ]
                    ),
                OneOf(
                        [
                            RandGaussianSmoothd(keys=["image"],prob=0.3),
                            RandGaussianSharpend(keys=["image"],prob=0.3),
                        ]
                    ),
                RandGibbsNoised(keys=["image"], prob=0.2)
            ]
        )
        
        # 局部裁剪transformer：pad-crop-旋转-强度偏移
        self.local_transforms = Compose(
            [
                SpatialPadd(keys="image", spatial_size=[self.local_crops_size, self.local_crops_size, self.local_crops_size]),
                RandSpatialCropSamplesd(
                    keys=["image"],
                    roi_size=[self.local_crops_size, self.local_crops_size, self.local_crops_size],
                    num_samples=self.local_crops_number,
                    random_center=True,
                    random_size=False,
                ),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[0]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[1]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[2]),
                # RandRotate90d(keys=["image"], prob=0.10, max_k=3,),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(1, 2)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 2)),
                OneOf(
                        [
                            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                            RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                            RandHistogramShiftd(keys=["image"], num_control_points=10, prob=0.3),
                        ]
                    ),
                RandGaussianSmoothd(keys=["image"], prob=0.3)
            ]
        )

        self.to_tensor = Compose(
            [
                ToTensord(keys=["image"])
            ]
        )
        
        self.global_transfo1 = Compose([self.global_transforms1, self.to_tensor])
        self.global_transfo2 = Compose([self.global_transforms2, self.to_tensor])
        self.local_transfo = Compose([self.local_transforms, self.to_tensor])

    def __call__(self, data):
        output = {}
        
        # global crops:
        # im_base = self.init_transforms(data)
        im_base = data
        global_crop_1 = self.global_transfo1(im_base)[0]["image"]
        global_crop_2 = self.global_transfo2(im_base)[0]["image"]

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]
        local_crops = [ each["image"]
             for each in self.local_transfo(im_base)
        ]
        # local crops:
        # local_crops = [
        #     self.local_transfo(im_base) for _ in range(self.local_crops_number)
        # ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output      



class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
