# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn

from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup_3d
import dinov2.utils.utils as dinov2_utils
import torch.nn as nn

from sam_med3d import build_sam3D_vit_b_ori

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep


def get_args_parser(
        description: Optional[str] = None,
        parents: Optional[List[argparse.ArgumentParser]] = None,
        add_help: bool = True,
    ):
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


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model_for_eval(config, pretrained_weights):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    try:
        dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    except FileNotFoundError as e:
        print(e)
        print('No weights found, using random initialization!')
    
    # debug打印模型参数key以及shape，并保存到txt
    # with open("/home/huangxinrui/Code/Project_Foundation_Model/3DINO/model_param_keys_and_shapes.txt", "w") as f:
    #     for k, v in model.state_dict().items():
    #         f.write(f"{k}: {tuple(v.shape)}\n")

    model.eval()
    model.cuda()
    return model


def setup_and_build_model_3d(args) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup_3d(args)
    model = build_model_for_eval(config, args.pretrained_weights)
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype



# 实例化sammed3d
## 定义一个SAMMed3d模型
class SAMMed3d(nn.Module):
    def __init__(self, a_model):
        super(SAMMed3d, self).__init__()
        self.model = a_model
        self.num_features = 768
    
    def forward(self, images):
        # 只对图像数据过image encoder
        image_embeddings = self.model.image_encoder(images)

        out_features = image_embeddings.mean(dim=[2, 3, 4])
        
        return out_features
    
    def get_intermediate_layers(self, images, n, return_class_token=False):
        """
        获取中间层特征
        :param images: 输入图像
        :param n: 中间层的索引
        :param return_class_token: 是否返回class token
        :return: 中间层特征
        """
        # 只对图像数据过image encoder
        image_embeddings = self.model.image_encoder.get_intermediate_layers(images)

        out_features = []
        for idx in n:
            out_features.append(image_embeddings[idx])
        
        return out_features



def setup_and_build_model_sammed3d(args):
    assert args.model_name=='sammed3d', 'The model is not sammed3d'

    cudnn.benchmark = True
    config = setup_3d(args)

    a_model = build_sam3D_vit_b_ori(pretrained=True, checkpoint_path=args.pretrained_weights)
    
    model = SAMMed3d(a_model)
    
    model.eval()
    model.cuda()
    return model, torch.float


# 实例化m3d
## 定义一个m3d模型
class M3D(nn.Module):
    def __init__(self, a_model):
        super(M3D, self).__init__()
        self.model = a_model
        self.num_features = 768
    
    def forward(self, images):
        ori_image_features = self.model.encode_image(images)

        cls_feature = ori_image_features[:, 0]
        
        return cls_feature

    def get_intermediate_layers(self, images, n, return_class_token=False):
        """
        获取中间层特征
        :param images: 输入图像
        :param n: 中间层的索引
        :param return_class_token: 是否返回class token
        :return: 中间层特征
        """
        image_size= images.shape[-1]

        # 只对图像数据过image encoder
        _, image_embeddings = self.model.vision_encoder(images)

        out_features = []
        for idx in n:
            feature = image_embeddings[idx][:,1:,:]  # 去掉cls token
            # 将2048变为8*16*16
            feature = feature.view(feature.shape[0], 8, 16, 16, -1)  # (B, 8, 16, 16, 768)
            # 空间维度插值为image_size//16*image_size//16*image_size//16，其他维度保持不变
            feature = torch.nn.functional.interpolate(
                feature.permute(0, 4, 1, 2, 3),
                size=(image_size//16, image_size//16, image_size//16),
                mode='trilinear',
                align_corners=False
            )
            # 变为 (B, D*H*W, C)
            feature = feature.permute(0, 2, 3, 4, 1).reshape(feature.shape[0], -1, feature.shape[1])
            out_features.append(feature)
        
        return out_features



def setup_and_build_model_m3d(args):
    assert args.model_name=='m3d', 'The model is not m3d'

    cudnn.benchmark = True
    config = setup_3d(args)

    a_model = AutoModel.from_pretrained(
        "GoodBaiBai88/M3D-CLIP",
        trust_remote_code=True
    )

    # a_model = AutoModel.from_pretrained(
    #     # "/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint/hub/models--GoodBaiBai88--M3D-CLIP/snapshots/ae091d89a0ef38b533ecc4ed21426f7658853963",
    #     "/home/huangxinrui/Code/Project_Foundation_Model/huggingface_test/checkpoint/hub/models--GoodBaiBai88--M3D-CLIP/snapshots/ae091d89a0ef38b533ecc4ed21426f7658853963",
    #     trust_remote_code=True
    # )
    
    model = M3D(a_model)
    
    model.eval()
    model.cuda()
    
    return model, torch.float



# 实例化swimunetr模型
## 定义一个swimunetr模型
class SwinUNETR(nn.Module):
    def __init__(self, a_model):
        super(SwinUNETR, self).__init__()
        self.model = a_model
        self.num_features = 48
    
    def forward(self, images):
        # 图像编码
        image_embeddings = self.model(images)

        # 对特征求avg pool
        hidden_embedding = image_embeddings[4]
        out_features = hidden_embedding.mean(dim=[2, 3, 4])
        
        return out_features
    
    # 此处修改，增加得到中间层特征
    def get_intermediate_layers(self, images):
        # 图像编码
        image_embeddings = self.model(images)
        
        return image_embeddings


def setup_and_build_model_swimunetr(args):
    assert args.model_name=='swimunetr', 'The model is not swimunetr'

    cudnn.benchmark = True
    config = setup_3d(args)

    # 定义SwinViT模型
    patch_size = ensure_tuple_rep(2, 3)
    window_size = ensure_tuple_rep(7, 3)
    a_model = SwinViT(
                in_chans=1,
                embed_dim=48,
                window_size=window_size,
                patch_size=patch_size,
                depths=[2, 2, 2, 2],
                num_heads=[3, 6, 12, 24],
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=torch.nn.LayerNorm,
                use_checkpoint=False,
                spatial_dims=3,
            )

    # 加载预训练权重
    model_dict = torch.load(args.pretrained_weights, map_location='cpu')
    # 修改key，将model_dict["state_dict"]中的"module."前缀去掉
    # 将包含“.mlp.fc”改为“.mlp.linear”
    new_state_dict = {}
    for k, v in model_dict["state_dict"].items():
        new_key = k.replace("module.", "")
        new_key = new_key.replace(".mlp.fc", ".mlp.linear")
        new_state_dict[new_key] = v
    msg = a_model.load_state_dict(new_state_dict, strict=False)
    print(msg)
    
    model = SwinUNETR(a_model)
    
    model.eval()
    model.cuda()
    
    return model, torch.float