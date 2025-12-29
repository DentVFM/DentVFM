# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
    EnsureTyped,
    Resized,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandSpatialCropd,
    CenterSpatialCropd,
    Identityd,
    OneOf,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    Lambdad,
    NormalizeIntensityd,
    Spacingd,
    SpatialPadd
)
from torchio.transforms import RandomAffine
import torch


def make_classification_transform_3d(dataset_name: str, image_size: int, min_int: float):
    """
    Create a training and validation transform for 3D classification tasks.

    Args:
        dataset_name: Name of the classification dataset (ICBM, COVID-CT-MD).
        image_size: Size of the image to be used for training.
        min_int: Minimum intensity value to map the image to.
    Returns:
        Training and validation transforms.
    """

    if image_size == 0:
        resize_transform = Identityd(keys=["image"])
    else:
        resize_transform = Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear")

    if dataset_name == 'ICBM':
        def label_map(x):
            if 20 <= x < 30:
                return 0
            elif 30 <= x < 40:
                return 1
            elif 40 <= x < 50:
                return 2
            elif 50 <= x <= 60:
                return 3

        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                resize_transform,
                OneOf(transforms=[
                    RandomAffine(include=["image"], p=0.3, degrees=(30, 30, 30),
                                 scales=(0.8, 1.25), translation=(0.1, 0.1, 0.1),
                                 default_pad_value=min_int),
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                Lambdad(keys=["label"], func=label_map)
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                resize_transform,
                Lambdad(keys=["label"], func=label_map)
            ]
        )

    elif dataset_name == 'COVID-CT-MD':
        def label_map(x):
            if x == 'Normal':
                return 0
            elif x == 'COVID-19':
                return 1
            elif x == 'Cap':
                return 2
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                   keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                Resized(keys=["image"], spatial_size=(144, 144, 112), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                # 修改
                # Lambdad(keys=["label"], func=label_map)
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                   keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                Resized(keys=["image"], spatial_size=(144, 144, 112), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
                # 修改
                # Lambdad(keys=["label"], func=label_map)
            ]
        )

    # 此处修改添加
    elif '9h_fracture_classification_ct_cbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[112, 112, 112]),
                Resized(keys=["image"], spatial_size=(112, 112, 112), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(96, 96, 96), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(96, 96, 96)),
            ]
        )
    
    elif '9h_orsplit_after_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(112, 112, 112), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(96, 96, 96), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(96, 96, 96)),
            ]
        )
    
    elif '9h_orsplit_before_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(112, 112, 112), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(96, 96, 96), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(96, 96, 96)),
            ]
        )
    
    elif '9h_single_double_after_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(112, 112, 112), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(96, 96, 96), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(96, 96, 96)),
            ]
        )
    
    elif '9h_single_double_before_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(112, 112, 112), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(96, 96, 96), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(96, 96, 96)),
            ]
        )
    
    elif '9h_tmj_abnormal_classification' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(112, 112, 112), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(96, 96, 96), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(96, 96, 96)),
            ]
        )



    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    return train_transforms, val_transforms


def make_classification_transform_3d_sammed3d(dataset_name: str, image_size: int, min_int: float):
    """
    Create a training and validation transform for 3D classification tasks.

    Args:
        dataset_name: Name of the classification dataset (ICBM, COVID-CT-MD).
        image_size: Size of the image to be used for training.
        min_int: Minimum intensity value to map the image to.
    Returns:
        Training and validation transforms.
    """

    if image_size == 0:
        resize_transform = Identityd(keys=["image"])
    else:
        resize_transform = Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear")

    if dataset_name == 'ICBM':
        def label_map(x):
            if 20 <= x < 30:
                return 0
            elif 30 <= x < 40:
                return 1
            elif 40 <= x < 50:
                return 2
            elif 50 <= x <= 60:
                return 3

        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                resize_transform,
                OneOf(transforms=[
                    RandomAffine(include=["image"], p=0.3, degrees=(30, 30, 30),
                                 scales=(0.8, 1.25), translation=(0.1, 0.1, 0.1),
                                 default_pad_value=min_int),
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                Lambdad(keys=["label"], func=label_map)
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                resize_transform,
                Lambdad(keys=["label"], func=label_map)
            ]
        )

    elif dataset_name == 'COVID-CT-MD':
        def label_map(x):
            if x == 'Normal':
                return 0
            elif x == 'COVID-19':
                return 1
            elif x == 'Cap':
                return 2
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                   keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                NormalizeIntensityd(
                    keys=["image"], nonzero=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                # 修改
                # Lambdad(keys=["label"], func=label_map)
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                   keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                NormalizeIntensityd(
                    keys=["image"], nonzero=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
                # 修改
                # Lambdad(keys=["label"], func=label_map)
            ]
        )
    
    # 此处修改添加
    elif '9h_fracture_classification_ct_cbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    
    elif '9h_orsplit_after_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    
    elif '9h_orsplit_before_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    
    elif '9h_single_double_after_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    
    elif '9h_single_double_before_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    
    elif '9h_tmj_abnormal_classification' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )



    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    return train_transforms, val_transforms


def make_classification_transform_3d_m3d(dataset_name: str, image_size: int, min_int: float):
    """
    Create a training and validation transform for 3D classification tasks.

    Args:
        dataset_name: Name of the classification dataset (ICBM, COVID-CT-MD).
        image_size: Size of the image to be used for training.
        min_int: Minimum intensity value to map the image to.
    Returns:
        Training and validation transforms.
    """

    if image_size == 0:
        resize_transform = Identityd(keys=["image"])
    else:
        resize_transform = Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear")

    if dataset_name == 'ICBM':
        def label_map(x):
            if 20 <= x < 30:
                return 0
            elif 30 <= x < 40:
                return 1
            elif 40 <= x < 50:
                return 2
            elif 50 <= x <= 60:
                return 3

        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                resize_transform,
                OneOf(transforms=[
                    RandomAffine(include=["image"], p=0.3, degrees=(30, 30, 30),
                                 scales=(0.8, 1.25), translation=(0.1, 0.1, 0.1),
                                 default_pad_value=min_int),
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                Lambdad(keys=["label"], func=label_map)
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                resize_transform,
                Lambdad(keys=["label"], func=label_map)
            ]
        )

    elif dataset_name == 'COVID-CT-MD':
        def label_map(x):
            if x == 'Normal':
                return 0
            elif x == 'COVID-19':
                return 1
            elif x == 'Cap':
                return 2
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                   keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                Resized(keys=["image"], spatial_size=(40, 260, 260), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(32, 256, 256), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                # 修改
                # Lambdad(keys=["label"], func=label_map)
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                   keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                Resized(keys=["image"], spatial_size=(40, 260, 260), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(32, 256, 256)),
                # 修改
                # Lambdad(keys=["label"], func=label_map)
            ]
        )
    
    # 此处修改添加
    elif '9h_fracture_classification_ct_cbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    
    elif '9h_orsplit_after_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    
    elif '9h_orsplit_before_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    
    elif '9h_single_double_after_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    
    elif '9h_single_double_before_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    
    elif '9h_tmj_abnormal_classification' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )




    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    return train_transforms, val_transforms


def make_classification_transform_3d_swimunetr(dataset_name: str, image_size: int, min_int: float):
    """
    Create a training and validation transform for 3D classification tasks.

    Args:
        dataset_name: Name of the classification dataset (ICBM, COVID-CT-MD).
        image_size: Size of the image to be used for training.
        min_int: Minimum intensity value to map the image to.
    Returns:
        Training and validation transforms.
    """

    if image_size == 0:
        resize_transform = Identityd(keys=["image"])
    else:
        resize_transform = Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear")

    if dataset_name == 'ICBM':
        def label_map(x):
            if 20 <= x < 30:
                return 0
            elif 30 <= x < 40:
                return 1
            elif 40 <= x < 50:
                return 2
            elif 50 <= x <= 60:
                return 3

        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                resize_transform,
                OneOf(transforms=[
                    RandomAffine(include=["image"], p=0.3, degrees=(30, 30, 30),
                                 scales=(0.8, 1.25), translation=(0.1, 0.1, 0.1),
                                 default_pad_value=min_int),
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                Lambdad(keys=["label"], func=label_map)
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                resize_transform,
                Lambdad(keys=["label"], func=label_map)
            ]
        )

    elif dataset_name == 'COVID-CT-MD':
        def label_map(x):
            if x == 'Normal':
                return 0
            elif x == 'COVID-19':
                return 1
            elif x == 'Cap':
                return 2
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                   keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                Resized(keys=["image"], spatial_size=(128, 128, 128), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(96, 96, 96), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                # 修改
                # Lambdad(keys=["label"], func=label_map)
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                   keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(96, 96, 96)),
                # 修改
                # Lambdad(keys=["label"], func=label_map)
            ]
        )
    
    # 此处修改添加
    elif '9h_fracture_classification_ct_cbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    
    elif '9h_orsplit_after_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    
    elif '9h_orsplit_before_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    
    elif '9h_single_double_after_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    
    elif '9h_single_double_before_ctcbct' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    
    elif '9h_tmj_abnormal_classification' in dataset_name:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[image_size, image_size, image_size]),
                Resized(keys=["image"], spatial_size=(144, 144, 144), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Lambdad(keys=["image"], func=lambda x:torch.nan_to_num(x, torch.nanmean(x).item())),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )




    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    return train_transforms, val_transforms