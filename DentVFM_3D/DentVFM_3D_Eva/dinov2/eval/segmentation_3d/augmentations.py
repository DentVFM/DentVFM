# Author: Tony Xu
#
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    MapTransform,
    EnsureTyped,
    RandSpatialCropSamplesd,
    RandScaleIntensityd,
    ConcatItemsd,
    DeleteItemsd,
    SpatialPadd,
    Lambdad
)
import torch
import numpy as np


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 3 is GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            result = []
            # merge label 1 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 1, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 3 is ET
            result.append(d[key] == 3)
            d[key] = torch.cat(result, dim=0).float()
        return d


# 此处修改，增加不同数据集用的transforms
def make_transforms(dataset_name, image_size, resize_scale, min_int):

    if dataset_name == 'BTCV':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    elif dataset_name == 'BraTS':

        train_transforms = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image1", "image2", "image3", "image4", "label"], ensure_channel_first=True),
                ConcatItemsd(keys=["image1", "image2", "image3", "image4"], name='image', dim=0),
                DeleteItemsd(keys=["image1", "image2", "image3", "image4"]),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandSpatialCropSamplesd(
                    keys=["image", "label"], num_samples=4, roi_size=(image_size, image_size, image_size), random_size=False
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image1", "image2", "image3", "image4", "label"], ensure_channel_first=True),
                ConcatItemsd(keys=["image1", "image2", "image3", "image4"], name='image', dim=0),
                DeleteItemsd(keys=["image1", "image2", "image3", "image4"]),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    elif dataset_name == 'LA-SEG':

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Lambdad(keys=["label"], func=lambda x: (x == 255).astype(np.uint8)),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 0.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Lambdad(keys=["label"], func=lambda x: (x == 255).astype(np.uint8)),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 0.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif dataset_name == 'TDSC-ABUS':

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    # 此处修改
    elif 'Dataset112_ToothFairy2' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96), value=0),
                SpatialPadd(keys=["label"], spatial_size=(96, 96, 96), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=3,
                    # neg=1,
                    neg=0,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96), value=0),
                SpatialPadd(keys=["label"], spatial_size=(96, 96, 96), value=0.),
            ]
        )
    
    elif 'Dataset113_NCseg' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=4,
                    # neg=1,
                    neg=0,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif 'Dataset114_Sts_miccia23' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96), value=0),
                SpatialPadd(keys=["label"], spatial_size=(96, 96, 96), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96), value=0),
                SpatialPadd(keys=["label"], spatial_size=(96, 96, 96), value=0.),
            ]
        )
    
    elif 'Dataset115_Pal' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96), value=0),
                SpatialPadd(keys=["label"], spatial_size=(96, 96, 96), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96), value=0),
                SpatialPadd(keys=["label"], spatial_size=(96, 96, 96), value=0.),
            ]
        )
    
    elif 'Dataset116_HeadNeck_miccai15' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96), value=0),
                SpatialPadd(keys=["label"], spatial_size=(96, 96, 96), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96), value=0),
                SpatialPadd(keys=["label"], spatial_size=(96, 96, 96), value=0.),
            ]
        )
    
    elif 'Dataset117_Pulpy3D' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96), value=0),
                SpatialPadd(keys=["label"], spatial_size=(96, 96, 96), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96), value=0),
                SpatialPadd(keys=["label"], spatial_size=(96, 96, 96), value=0.),
            ]
        )
    
    elif 'Dataset118_ToothFairy3' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96), value=0),
                SpatialPadd(keys=["label"], spatial_size=(96, 96, 96), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=4,
                    # neg=1,
                    neg=0,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96), value=0),
                SpatialPadd(keys=["label"], spatial_size=(96, 96, 96), value=0.),
            ]
        )


    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # Flatten transforms to allow caching of non-random transforms if needed
    train_transforms = train_transforms.flatten()
    val_transforms = val_transforms.flatten()

    return train_transforms, val_transforms


def make_transforms_sammed3d(dataset_name, image_size, resize_scale, min_int):

    if dataset_name == 'BTCV':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    elif dataset_name == 'BraTS':

        train_transforms = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image1", "image2", "image3", "image4", "label"], ensure_channel_first=True),
                ConcatItemsd(keys=["image1", "image2", "image3", "image4"], name='image', dim=0),
                DeleteItemsd(keys=["image1", "image2", "image3", "image4"]),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandSpatialCropSamplesd(
                    keys=["image", "label"], num_samples=4, roi_size=(image_size, image_size, image_size), random_size=False
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image1", "image2", "image3", "image4", "label"], ensure_channel_first=True),
                ConcatItemsd(keys=["image1", "image2", "image3", "image4"], name='image', dim=0),
                DeleteItemsd(keys=["image1", "image2", "image3", "image4"]),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    elif dataset_name == 'LA-SEG':

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Lambdad(keys=["label"], func=lambda x: (x == 255).astype(np.uint8)),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 0.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Lambdad(keys=["label"], func=lambda x: (x == 255).astype(np.uint8)),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 0.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif dataset_name == 'TDSC-ABUS':

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )


    # 此处修改
    elif 'Dataset112_ToothFairy2' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=3,
                    # neg=1,
                    neg=0,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    
    elif 'Dataset113_NCseg' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif 'Dataset114_Sts_miccia23' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    
    elif 'Dataset115_Pal' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    
    elif 'Dataset116_HeadNeck_miccai15' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    
    elif 'Dataset117_Pulpy3D' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    
    elif 'Dataset118_ToothFairy3' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=4,
                    # neg=1,
                    neg=0,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )



    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # Flatten transforms to allow caching of non-random transforms if needed
    train_transforms = train_transforms.flatten()
    val_transforms = val_transforms.flatten()

    return train_transforms, val_transforms


def make_transforms_m3d(dataset_name, image_size, resize_scale, min_int):

    if dataset_name == 'BTCV':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    elif dataset_name == 'BraTS':

        train_transforms = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image1", "image2", "image3", "image4", "label"], ensure_channel_first=True),
                ConcatItemsd(keys=["image1", "image2", "image3", "image4"], name='image', dim=0),
                DeleteItemsd(keys=["image1", "image2", "image3", "image4"]),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandSpatialCropSamplesd(
                    keys=["image", "label"], num_samples=4, roi_size=(image_size, image_size, image_size), random_size=False
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image1", "image2", "image3", "image4", "label"], ensure_channel_first=True),
                ConcatItemsd(keys=["image1", "image2", "image3", "image4"], name='image', dim=0),
                DeleteItemsd(keys=["image1", "image2", "image3", "image4"]),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    elif dataset_name == 'LA-SEG':

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Lambdad(keys=["label"], func=lambda x: (x == 255).astype(np.uint8)),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 0.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Lambdad(keys=["label"], func=lambda x: (x == 255).astype(np.uint8)),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 0.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif dataset_name == 'TDSC-ABUS':

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    # 此处修改
    elif 'Dataset112_ToothFairy2' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=3,
                    # neg=1,
                    neg=0,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    
    elif 'Dataset113_NCseg' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif 'Dataset114_Sts_miccia23' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    
    elif 'Dataset115_Pal' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    
    elif 'Dataset116_HeadNeck_miccai15' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    
    elif 'Dataset117_Pulpy3D' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    
    elif 'Dataset118_ToothFairy3' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=4,
                    # neg=1,
                    neg=0,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )


    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # Flatten transforms to allow caching of non-random transforms if needed
    train_transforms = train_transforms.flatten()
    val_transforms = val_transforms.flatten()

    return train_transforms, val_transforms


def make_transforms_swimunetr(dataset_name, image_size, resize_scale, min_int):

    if dataset_name == 'BTCV':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    elif dataset_name == 'BraTS':

        train_transforms = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image1", "image2", "image3", "image4", "label"], ensure_channel_first=True),
                ConcatItemsd(keys=["image1", "image2", "image3", "image4"], name='image', dim=0),
                DeleteItemsd(keys=["image1", "image2", "image3", "image4"]),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandSpatialCropSamplesd(
                    keys=["image", "label"], num_samples=4, roi_size=(image_size, image_size, image_size), random_size=False
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image1", "image2", "image3", "image4", "label"], ensure_channel_first=True),
                ConcatItemsd(keys=["image1", "image2", "image3", "image4"], name='image', dim=0),
                DeleteItemsd(keys=["image1", "image2", "image3", "image4"]),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    elif dataset_name == 'LA-SEG':

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Lambdad(keys=["label"], func=lambda x: (x == 255).astype(np.uint8)),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 0.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Lambdad(keys=["label"], func=lambda x: (x == 255).astype(np.uint8)),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 0.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif dataset_name == 'TDSC-ABUS':

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )


    # 此处修改
    elif 'Dataset112_ToothFairy2' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=3,
                    # neg=1,
                    neg=0,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    
    elif 'Dataset113_NCseg' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif 'Dataset114_Sts_miccia23' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    
    elif 'Dataset115_Pal' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    
    elif 'Dataset116_HeadNeck_miccai15' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    
    elif 'Dataset117_Pulpy3D' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    
    elif 'Dataset118_ToothFairy3' in dataset_name:

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=4,
                    # neg=1,
                    neg=0,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=0, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=0),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )



    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # Flatten transforms to allow caching of non-random transforms if needed
    train_transforms = train_transforms.flatten()
    val_transforms = val_transforms.flatten()

    return train_transforms, val_transforms