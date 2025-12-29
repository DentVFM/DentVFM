# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Union

import torch
import torch.nn.functional as F
from .utils import _DINOV2_BASE_URL, _make_dinov2_model_name


class Weights(Enum):
    LVD142M = "LVD142M"


# def _make_dinov2_model(
#     *,
#     arch_name: str = "vit_large",
#     img_size: int = 518,
#     patch_size: int = 14,
#     init_values: float = 1.0,
#     ffn_layer: str = "mlp",
#     block_chunks: int = 0,
#     num_register_tokens: int = 0,
#     interpolate_antialias: bool = False,
#     interpolate_offset: float = 0.1,
#     pretrained: bool = True,
#     weights: Union[Weights, str] = Weights.LVD142M,
#     **kwargs,
# ):
#     from . import vision_transformer as vits

#     if isinstance(weights, str):
#         try:
#             weights = Weights[weights]
#         except KeyError:
#             raise AssertionError(f"Unsupported weights: {weights}")

#     model_base_name = _make_dinov2_model_name(arch_name, patch_size)
#     vit_kwargs = dict(
#         img_size=img_size,
#         patch_size=patch_size,
#         init_values=init_values,
#         ffn_layer=ffn_layer,
#         block_chunks=block_chunks,
#         num_register_tokens=num_register_tokens,
#         interpolate_antialias=interpolate_antialias,
#         interpolate_offset=interpolate_offset,
#     )
#     vit_kwargs.update(**kwargs)
#     model = vits.__dict__[arch_name](**vit_kwargs)

#     if pretrained:
#         model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
#         url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
#         state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
#         # import ipdb; ipdb.set_trace()
#         # pos_embed = F.interpolate(state_dict['pos_embed'][:, 1:, :].permute(0,2,1), size=32*32, mode='linear', align_corners=False).permute(0,2,1)
#         # state_dict['pos_embed'] = torch.cat((state_dict['pos_embed'][:, 0, :].unsqueeze(0), pos_embed), dim=1)
#         # state_dict['patch_embed.proj.weight'] = F.interpolate(state_dict['patch_embed.proj.weight'], size=(16, 16), mode='bilinear', align_corners=False)
#         model.load_state_dict(state_dict, strict=True)

#     return model


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD142M,
    **kwargs,
):
    from . import vision_transformer as vits
    

    if isinstance(weights, str) and not weights.endswith(".pth"):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    model_base_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=16,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained and isinstance(weights, str) and weights.endswith(".pth"):
        state_dict = torch.load(weights)
        pos_embed = F.interpolate(state_dict['pos_embed'][:, 1:, :].permute(0,2,1), size=32*32, mode='linear', align_corners=False).permute(0,2,1)
        state_dict['pos_embed'] = torch.cat((state_dict['pos_embed'][:, 0, :].unsqueeze(0), pos_embed), dim=1)
        state_dict['patch_embed.proj.weight'] = F.interpolate(state_dict['patch_embed.proj.weight'], size=(16, 16), mode='bilinear', align_corners=False)
        model.load_state_dict(state_dict, strict=True)
    elif pretrained:
        model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
        url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        # import ipdb; ipdb.set_trace()
        pos_embed = F.interpolate(state_dict['pos_embed'][:, 1:, :].permute(0,2,1), size=32*32, mode='linear', align_corners=False).permute(0,2,1)
        state_dict['pos_embed'] = torch.cat((state_dict['pos_embed'][:, 0, :].unsqueeze(0), pos_embed), dim=1)
        state_dict['patch_embed.proj.weight'] = F.interpolate(state_dict['patch_embed.proj.weight'], size=(16, 16), mode='bilinear', align_corners=False)
        model.load_state_dict(state_dict, strict=True)
        

    return model


def dinov2_vits14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_small", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitb14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-B/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_base", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitl14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_large", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitg14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-g/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        **kwargs,
    )


def dinov2_vits14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-S/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_small",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitb14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-B/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_base",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitl14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-L/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_large",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitg14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-g/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )

def dinov2_vitb14_dentvfm(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    # weights = '/home/xiaofan/code/carotis_plaque/code/dentvfm_pretrained/Fm_2d_Test0011-4/eval/training_74999/teacher_checkpoint_backbone_ori.pth'
    # weights = '/home/xiaofan/code/carotis_plaque/code/dentvfm_pretrained/Fm_2d_Test0011-4/eval/training_87499/teacher_checkpoint_backbone_ori.pth'
    # weights = '/home/xiaofan/code/carotis_plaque/code/dentvfm_pretrained/Fm_2d_Test0011-4/eval/training_62499/teacher_checkpoint_backbone_ori.pth'
    # weights = '/home/huangxinrui/Code/Project_VFM/cp_code/cp_code/my_checkpoints/mask_rcnn_split_our/Fm_2d_Test0011-4/eval/training_74999/teacher_checkpoint_backbone_ori.pth'
    weights = '/home/huangxinrui/Code/Project_VFM/cp_code/cp_code/pretrained_models/Fm_2d_Test0011-4/eval/training_87499/teacher_checkpoint_backbone_ori.pth'
    new_kwargs = dict(
        img_size = 518,
        patch_size = 14,
        ffn_layer = 'swiglufused', #'swiglu' 'mlp'
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        drop_path_rate=0.3,
        # init_values=1.0e-05,
        drop_path_uniform=True
    )
    kwargs.update(new_kwargs)
    """
    DINOv2 ViT-B/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_base",
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )