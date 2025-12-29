# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn

from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup
import dinov2.utils.utils as dinov2_utils

import torch.nn as nn
import torchvision.models as models

from open_clip import create_model_from_pretrained, get_tokenizer
from transformers import CLIPProcessor, CLIPModel, SamModel, SamProcessor, AutoProcessor
from lvm_vit_model import vit_encoder_b, load_weight_for_vit_encoder
from sammed2d_vit_model import ImageEncoderViT
from functools import partial

# 设定eval时基础参数
def get_args_parser(
        description: Optional[str] = None,
        parents: Optional[List[argparse.ArgumentParser]] = None,
        add_help: bool = True,
    ):
    '''
    预定义一些参数

    配置文件、模型权重、eval结果输出路径、其他等
    '''
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    return parser

# 获取数据类型
def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float

# 实例化模型，并加载模型权重
def build_model_for_eval(config, pretrained_weights):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model

# 读取配置，并加载预训练模型
def setup_and_build_model(args) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    # 读取配置并设置
    config = setup(args)
    # 实例化并加载模型权重
    model = build_model_for_eval(config, args.pretrained_weights)
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype


# 新增加载dinov3参数
def setup_and_build_model_dinov3(args):
    cudnn.benchmark = True
    # 读取配置并设置
    config = setup(args)
    # 实例化并加载模型权重
    REPO_DIR = "/home/jovyan/dataset/Code/Project_Foundation_Model/dinov3-main"
    weights_path = args.pretrained_weights
    print(f'dinov3_{args.model_version_name}')
    print(weights_path)
    model = torch.hub.load(REPO_DIR, f'dinov3_{args.model_version_name}', source='local', weights=weights_path)
    autocast_dtype = torch.float
    model.eval()
    model.cuda()
    return model, autocast_dtype


# 加载预训练resnet50
def setup_and_build_model_resnet50(args):
    assert args.model_name=='resnet50', 'The model is not resnet50'
    
    cudnn.benchmark = True
    config = setup(args)
    model = models.resnet50(pretrained=False)
    state_dict = torch.load(args.pretrained_weights)
    model.load_state_dict(state_dict)
    # 保留backbone
    model.fc = nn.Identity()
    autocast_dtype = torch.float
    model.eval()
    model.cuda()
    return model, autocast_dtype


# 加载预训练lvmresnet50
def setup_and_build_model_lvm_resnet50(args):
    assert args.model_name=='lvm_resnet50', 'The model is not lvm_resnet50'
    
    cudnn.benchmark = True
    config = setup(args)
    model = models.resnet50(pretrained=False)
    state_dict = torch.load(args.pretrained_weights)
    msg = model.load_state_dict(state_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(args.pretrained_weights, msg))
    # 保留backbone
    model.fc = nn.Identity()
    autocast_dtype = torch.float
    model.eval()
    model.cuda()
    return model, autocast_dtype


# 加载预训练的lvmvitb16
def setup_and_build_model_lvm_vitb16(args):
    assert args.model_name=='lvm_vitb16', 'The model is not lvm_vitb16'

    cudnn.benchmark = True
    config = setup(args)

    model = vit_encoder_b()
    pretrained_weight = load_weight_for_vit_encoder(args.pretrained_weights)
    msg = model.load_state_dict(pretrained_weight, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(args.pretrained_weights, msg))

    autocast_dtype = torch.float
    model.eval()
    model.cuda()
    return model, autocast_dtype


# 加载预训练biomedclip
## 定义一个biomedclip模型
class Biomedclip(nn.Module):
    def __init__(self, a_model):
        super(Biomedclip, self).__init__()
        self.model = a_model
    
    def forward(self, images):
        image_features = self.model.encode_image(images)
        
        return image_features
## 实例化biomedclip模型
def setup_and_build_model_biomedclip(args):
    assert args.model_name=='biomedclip', 'The model is not biomedclip'
    
    cudnn.benchmark = True
    config = setup(args)
    a_model, a_preprocess = create_model_from_pretrained(args.model_version_name)
    model = Biomedclip(a_model)
    model.eval()
    model.cuda()
    return model, torch.float, a_preprocess


# 加载预训练clip
## 定义一个clip模型
class Clip(nn.Module):
    def __init__(self, a_model):
        super(Clip, self).__init__()
        self.model = a_model

    def forward(self, images):
        image_features = self.model.get_image_features(images)
        
        return image_features
## 实例化clip模型
def setup_and_build_model_clip(args):
    assert args.model_name=='clip', 'The model is not clip'
    
    cudnn.benchmark = True
    config = setup(args)
    a_model = CLIPModel.from_pretrained(args.model_version_name)
    a_preprocess = AutoProcessor.from_pretrained(args.model_version_name)
    model = Clip(a_model)
    model.eval()
    model.cuda()
    return model, torch.float, a_preprocess


# 加载预训练SAM
## 定义一个SAM模型
class SAM(nn.Module):
    def __init__(self, a_model):
        super(SAM, self).__init__()
        self.model = a_model
    
    def forward(self, images):
        image_features = self.model.get_image_embeddings(images)
        out_features = image_features.mean(dim=[2, 3])
        
        return out_features
## 实例化SAM模型
def setup_and_build_model_sam(args):
    assert args.model_name=='sam' or args.model_name=='medsam', 'The model is not sam'
    
    cudnn.benchmark = True
    config = setup(args)
    a_model = SamModel.from_pretrained(args.model_version_name)
    a_preprocess = SamProcessor.from_pretrained(args.model_version_name)
    model = SAM(a_model)
    model.eval()
    model.cuda()
    return model, torch.float, a_preprocess


# 加载SAM-Med2D模型
## 定义一个SAMMed2d模型
class SAMMed2d(nn.Module):
    def __init__(self, a_model):
        super(SAMMed2d, self).__init__()
        self.model = a_model
    
    def forward(self, images):
        image_features = self.model(images)
        out_features = image_features.mean(dim=[2, 3])
        
        return out_features
def setup_and_build_model_sammed2d(args):
    assert args.model_name=='sammed2d', 'The model is not sammed2d'

    cudnn.benchmark = True
    config = setup(args)
    # 实例图像编码器
    a_model = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=256,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos = True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
            adapter_train = True,
        )
    # 加载模型参数
    image_encoder_state_dict = {}
    with open(args.pretrained_weights, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
        for key,value in state_dict['model'].items():
            if 'image_encoder.' in key:
                image_encoder_state_dict[key.replace('image_encoder.','')] = value
    msg = a_model.load_state_dict(image_encoder_state_dict, False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(args.pretrained_weights, msg))
    model = SAMMed2d(a_model)
    model.eval()
    model.cuda()
    return model, torch.float
