# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

## 原版数据增强

import logging

from torchvision import transforms
from monai.transforms import (
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
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)


logger = logging.getLogger("dinov2")
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
        
        self.init_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(self.preprocessing.space_x, self.preprocessing.space_y, self.preprocessing.space_z), mode=("bilinear")),
                ScaleIntensityRanged(
                    keys=["image"], a_min=self.preprocessing.a_min, a_max=self.preprocessing.a_max, b_min=self.preprocessing.b_min, b_max=self.preprocessing.b_max, clip=True
                ),
                CropForegroundd(keys=["image"], source_key="image"),
            ]
        )
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
                RandRotate90d(keys=["image"], prob=0.10, max_k=3,),
                RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20,),
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
                RandRotate90d(keys=["image"], prob=0.10, max_k=3,),
                RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20,),
            ]
        )
        
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
                RandRotate90d(keys=["image"], prob=0.10, max_k=3,),
                RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20,),
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
