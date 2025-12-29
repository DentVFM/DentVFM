# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from mmcv.runner import BaseModule
from mmcv.ops import DeformConv2d
from mmdet.models.builder import BACKBONES
import torch.distributed as dist

import torch.nn.functional as F
import torch.nn as nn

from functools import partial
import itertools
import math
import copy
###Hack torch to load DINOv2####
from torch import Tensor
import torch._tensor
try:
    torch._tensor._rebuild_from_type_v2
except AttributeError:
    def _rebuild_from_type_v2(func, new_type, args, state):
        ret = func(*args)
        if type(ret) is not new_type:
            ret = ret.as_subclass(new_type)
        # Tensor does define __setstate__ even though it doesn't define
        # __getstate__. So only use __setstate__ if it is NOT the one defined
        # on Tensor
        if (
            getattr(ret.__class__, "__setstate__", Tensor.__setstate__)
            is not Tensor.__setstate__
        ):
            ret.__setstate__(state)
        else:
            ret = torch._utils._set_obj_state(ret, state)
        return ret

    torch._tensor._rebuild_from_type_v2 = _rebuild_from_type_v2

import torch._utils
try:
    torch._utils._set_obj_state
except AttributeError:
    def _set_obj_state(obj, state):
        if isinstance(state, tuple):
            if not len(state) == 2:
                raise RuntimeError(f"Invalid serialized state: {state}")
            dict_state = state[0]
            slots_state = state[1]
        else:
            dict_state = state
            slots_state = None

        # Starting with Python 3.11, the __dict__ attribute is lazily created
        # and is serialized as None when not needed.
        if dict_state:
            for k, v in dict_state.items():
                setattr(obj, k, v)
    
        if slots_state:
            for k, v in slots_state.items():
                setattr(obj, k, v)
        return obj

    torch._utils._set_obj_state = _set_obj_state
##Hack torch to load DINOv2####

##Hack DINOv2 ##
from .dinov2.hack_dinov2 import initialize_model
# from .dinov2.vision_transoformer import  
##Hack DINOv2 ##

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    # @torch.inference_mode()
    def forward(self, x):
        pads = list(
            itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1])
        )
        output = F.pad(x, pads)
        return output


class DeformableConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DeformableConvNet, self).__init__()
        self.offsets = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=padding)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
   
    def forward(self, x):
        offsets = self.offsets(x)
        x = self.deform_conv(x, offsets)
        return x


class LearnableGate(torch.nn.Module):
    def __init__(self, n, k, out_num, temperature):
        super(LearnableGate, self).__init__()
        self.n = n
        self.k = k
        self.out_num = out_num
        self.temperature = temperature
        self.scores = torch.nn.Parameter(torch.randn(n, out_num))
        torch.nn.init.uniform_(self.scores, a=0, b=1)

    def forward(self, X):
        """
        Args:
            X: (B, n, D) Input features with batch size B, number of features per sample n, and dimension of each feature D.
        Returns:
            gates: (B, n, k) STE outputs for each channel (forward pass is discrete k-hot; backward pass gradients come from the soft distribution).
        """
        B, n, D, _, _ = X.shape
        assert n == self.n, "Input feature dimension n must match the initialized n"
        
        scores = self.scores.unsqueeze(0).expand(B, -1, -1)  # shape: (B, n, k)
        soft_scores = F.softmax(scores / self.temperature, dim=1)  # shape: (B, n, k)
        topk_indices = torch.topk(soft_scores, self.k, dim=1).indices  # shape: (B, k, k)
        
        # Create a zero tensor for the hard selections with shape (B, n, k)
        sparse_scores = torch.zeros_like(soft_scores)  # shape: (B, n, k)
        # Create batch indices with shape (B, k, out)
        batch_idx = torch.arange(B, device=X.device).view(B, 1, 1).expand(B, self.k, self.out_num)
        # Create channel indices with shape (B, k, out)
        channel_idx = torch.arange(self.out_num, device=X.device).view(1, 1, self.out_num).expand(B, self.k, self.out_num)
        sparse_scores[batch_idx, topk_indices, channel_idx] = soft_scores[batch_idx, topk_indices, channel_idx] 

        gates = sparse_scores - scores.detach() + scores  # shape: (B, n, k)
        gates = gates / gates.sum(dim=1, keepdim=True)
        return gates


@BACKBONES.register_module()
class DINOViTSplitFusion(BaseModule):
    """DINO Vision Transformer Split version."""

    def __init__(
        self,
        backbone_size="small",
        register_version=False,
        tune_register=True,
        out_indices=[8,9,10,11],
        select_layers=[10,11],
        channels=384,
        tuning_type="frozen",
        output_orgimg=False,
        drop_path_rate=0,
        drop_path_uniform=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.output_orgimg = output_orgimg
        self.register_version = register_version
        # load dinov2 from torch.hub
        if not register_version:
            backbone_archs = {
                'small': 'vits14',
                'base': 'vitb14',
                'large': 'vitl14',
                'giant': 'vitg14',
                'base_dentvfm': 'vitb14_dentvfm'
            }
        else:
            backbone_archs = {
                'small': 'vits14_reg',
                'base': 'vitb14_reg',
                'large': 'vitl14_reg',
                'giant': 'vitg14_reg',
                'base_dentvfm': 'vitb14_dentvfm'
            }     
        backbone_arch = backbone_archs[backbone_size]
        # import ipdb; ipdb.set_trace()
        self.channels = channels
        backbone_name = f'dinov2_{backbone_arch}'
        # torch.hub._validate_not_a_forked_repo=lambda a, b, c: True #modify a bug of torch 1.x for loading hub weights
        # backbone_model = torch.hub.load(
        #     repo_or_dir = "facebookresearch/dinov2", model=backbone_name
        # )
        backbone_model = initialize_model(backbone_name=backbone_arch, patch_size=14, drop_path_rate=drop_path_rate, drop_path_uniform=drop_path_uniform)

        if register_version:
            self.num_register_tokens = backbone_model.num_register_tokens

        # Selecting layers
        self.select_layers = select_layers
        split_head = []
        for layer_id in self.select_layers:
            copy_blk = copy.deepcopy(backbone_model.blocks[layer_id])
            split_head.append(copy_blk)
        self.split_head = nn.Sequential(*split_head)

        # Hook the forward features
        self.split_activations = None
        backbone_model.blocks[self.select_layers[0]-1].register_forward_hook(self.get_activation)

        # initialize backbone model
        backbone_model.forward = partial(
            backbone_model.get_intermediate_layers,
            n = out_indices,
            reshape = True,
            norm = True
        )
        self.patch_size = backbone_model.patch_embed.patch_size[0]
        backbone_model.register_forward_pre_hook(
            lambda _, x: CenterPadding(self.patch_size)(x[0])
        )
        # Open the gradient of the given layer
        if tuning_type == "frozen":
            for param in backbone_model.parameters():
                param.requires_grad = False

        elif tuning_type == "all":
            for param in backbone_model.parameters():
                param.requires_grad = True

        elif isinstance(tuning_type, list):
            for param in backbone_model.parameters():
                param.requires_grad = False

            for layer_id in tuning_type:
                for param in backbone_model.blocks[layer_id].parameters():
                    param.requires_grad = True
        else:
            raise AttributeError(f"{tuning_type} is not supported !!!!")
            
        if register_version and tune_register:
            # import ipdb; ipdb.set_trace()
            for param in backbone_model.register_tokens:
                param.requires_grad = True

        self.backbone = backbone_model
        frozen_out_dim = channels
        self.frozen_conv = nn.Sequential(*[
            nn.Conv2d(in_channels=channels*len(out_indices), out_channels=channels, kernel_size=1, padding=0),
            nn.GELU(),
            DeformableConvNet(in_channels=channels, out_channels=frozen_out_dim, kernel_size=3, padding=1),
            nn.GELU()
        ])
        self.fusion_net = nn.Sequential(*[
            nn.Conv2d(in_channels=channels+frozen_out_dim, out_channels=channels, kernel_size=1, padding=0),
            nn.GELU(),
            DeformableConvNet(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.GELU()
        ])

        self.up1 = nn.Sequential(*[
            nn.ConvTranspose2d(channels, channels, 2, 2),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            nn.ConvTranspose2d(channels, channels, 2, 2),
            nn.GroupNorm(32, channels),
            nn.GELU(),
        ])
        self.up2 = nn.Sequential(*[
            nn.ConvTranspose2d(channels, channels, 2, 2),
            nn.GroupNorm(32, channels),
            nn.GELU()
        ])
        self.up3 = nn.Identity()
        self.up4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.norm1 = nn.SyncBatchNorm(channels)
        self.norm2 = nn.SyncBatchNorm(channels)
        self.norm3 = nn.SyncBatchNorm(channels)
        self.norm4 = nn.SyncBatchNorm(channels)

    @property
    def get_activation(self):
        def hook(model, input, output):
            self.split_activations = output.detach()
            # import ipdb; ipdb.set_trace()
            # self.split_input_features = input[0].detach()
        return hook
    
    def reshape_vit_tokens(self, x, norm=True):
        """
        reshape vit tokens from (b, L, D) to (b, D, h, w), refering to dinov2/model/vision_transformers.py "get_intermediate_layers" function
        input: x(b, L, D)
        output: x_(b, D, h, w)
        """
        b, L, D = x.shape

        if norm:
            x = self.backbone.norm(x)
        x_ = x[:, 1:, :] # drop class tokens in the batch
        if self.register_version:
            x_ = x_[:, self.num_register_tokens:, :] # drop register tokens in the batch
        x_ = x_.reshape(b, self.h, self.w, -1).permute(0, 3, 1, 2).contiguous()
        return x_

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        frozen_features = self.backbone(x)
        b, _, H, W = x.shape
        self.h, self.w = math.ceil(H/self.patch_size), math.ceil(W/self.patch_size)
        # split head features
        tuned_features = self.split_head(self.split_activations)
        # tuned_features = self.split_head(self.split_input_features)
        tuned_features = self.reshape_vit_tokens(tuned_features) # b,D,h,w
        # tuned_features = self.norm1(self.up1(tuned_features))

        frozen_features = torch.cat(frozen_features, dim=1) # b,nD,h,w
        frozen_features = self.frozen_conv(frozen_features) # b,D,h,w
        # frozen_features = self.norm2(self.up2(frozen_features))
        
        # return [tuned_features, frozen_features]
        x = torch.cat([frozen_features, tuned_features], dim=1) # b,2D,h,w
        x = self.fusion_net(x) # b,D,h,w
        #############################

        x1 = self.norm1(self.up1(x))
        x2 = self.norm2(self.up2(x))
        x3 = self.norm3(self.up3(x))
        x4 = self.norm4(self.up4(x))

        return [x1, x2, x3, x4]


@BACKBONES.register_module()
class DINOViTSplitLearnableGate(BaseModule):
    """DINO Vision Transformer Split with learnable gate."""

    def __init__(
        self,
        backbone_size="small",
        register_version=False,
        tune_register=True,
        channels=384,
        out_num=4,
        select_layers=[10,11],
        tuning_type="frozen",
        output_orgimg=False,
        freeze_learnable_gate=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.iters = 1
        self.out_num = out_num
        self.output_orgimg = output_orgimg
        self.register_version = register_version
        # load dinov2 from torch.hub
        if not register_version:
            backbone_archs = {
                'small': 'vits14',
                'base': 'vitb14',
                'large': 'vitl14',
                'giant': 'vitg14',
            }
        else:
            backbone_archs = {
                'small': 'vits14_reg',
                'base': 'vitb14_reg',
                'large': 'vitl14_reg',
                'giant': 'vitg14_reg',
            }     
        backbone_arch = backbone_archs[backbone_size]
        # import ipdb; ipdb.set_trace()
        backbone_name = f'dinov2_{backbone_arch}'
        torch.hub._validate_not_a_forked_repo=lambda a, b, c: True #modify a bug of torch 1.x for loading hub weights
        backbone_model = torch.hub.load(
            repo_or_dir = "facebookresearch/dinov2", model=backbone_name
        )
        if register_version:
            self.num_register_tokens = backbone_model.num_register_tokens
        # import ipdb; ipdb.set_trace()
        # Selecting layers
        self.select_layers = select_layers
        split_head = []
        for layer_id in self.select_layers:
            copy_blk = copy.deepcopy(backbone_model.blocks[layer_id])
            split_head.append(copy_blk)
        self.split_head = nn.Sequential(*split_head)

        # Hook the forward features
        self.split_activations = None
        backbone_model.blocks[self.select_layers[0]-1].register_forward_hook(self.get_activation)

        # initialize backbone model
        blk_num = len(backbone_model.blocks)
        backbone_model.forward = partial(
            backbone_model.get_intermediate_layers,
            n = [i for i in range(blk_num)],
            reshape = True,
            norm = True,
            return_class_token = False
        )

        self.learnable_gate = LearnableGate(n=blk_num, k=out_num)
        frozen_out_dim = channels
        self.frozen_conv = nn.Sequential(*[
            nn.Conv2d(in_channels=channels*self.out_num, out_channels=channels, kernel_size=1, padding=0),
            nn.GELU(),
            DeformableConvNet(in_channels=channels, out_channels=frozen_out_dim, kernel_size=3, padding=1),
            nn.GELU()
        ])
        self.fusion_net = nn.Sequential(*[
            nn.Conv2d(in_channels=channels+frozen_out_dim, out_channels=channels, kernel_size=1, padding=0),
            nn.GELU(),
            DeformableConvNet(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.GELU()
        ])

        self.patch_size = backbone_model.patch_embed.patch_size[0]
        backbone_model.register_forward_pre_hook(
            lambda _, x: CenterPadding(self.patch_size)(x[0])
        )
        # Open the gradient of the given layer
        if tuning_type == "frozen":
            for param in backbone_model.parameters():
                param.requires_grad = False

        elif tuning_type == "all":
            for param in backbone_model.parameters():
                param.requires_grad = True

        elif isinstance(tuning_type, list):
            for param in backbone_model.parameters():
                param.requires_grad = False

            for layer_id in tuning_type:
                for param in backbone_model.blocks[layer_id].parameters():
                    param.requires_grad = True
        else:
            raise AttributeError(f"{tuning_type} is not supported !!!!")
        # import ipdb; ipdb.set_trace()
        if freeze_learnable_gate:
            for param in self.learnable_gate.parameters():
                param.requires_grad = False

        if register_version and tune_register:
            # import ipdb; ipdb.set_trace()
            for param in backbone_model.register_tokens:
                param.requires_grad = True

        self.up1 = nn.Sequential(*[
            nn.ConvTranspose2d(channels, channels, 2, 2),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            nn.ConvTranspose2d(channels, channels, 2, 2),
            nn.GroupNorm(32, channels),
            nn.GELU(),
        ])
        self.up2 = nn.Sequential(*[
            nn.ConvTranspose2d(channels, channels, 2, 2),
            nn.GroupNorm(32, channels),
            nn.GELU()
        ])
        self.up3 = nn.Identity()
        self.up4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.norm1 = nn.SyncBatchNorm(channels)
        self.norm2 = nn.SyncBatchNorm(channels)
        self.norm3 = nn.SyncBatchNorm(channels)
        self.norm4 = nn.SyncBatchNorm(channels)

        self.backbone = backbone_model
    
    @property
    def get_activation(self):
        def hook(model, input, output):
            self.split_activations = output.detach()
        return hook
    
    def reshape_vit_tokens(self, x, norm=True):
        """
        reshape vit tokens from (b, L, D) to (b, D, h, w), refering to dinov2/model/vision_transformers.py "get_intermediate_layers" function
        input: x(b, L, D)
        output: x_(b, D, h, w)
        """
        b, L, D = x.shape

        if norm:
            x = self.backbone.norm(x)
        x_ = x[:, 1:, :] # drop class tokens in the batch
        if self.register_version:
            x_ = x_[:, self.num_register_tokens:, :] # drop register tokens in the batch
        # resized_len = int(math.sqrt(L-1))
        x_ = x_.reshape(b, self.h, self.w, -1).permute(0, 3, 1, 2).contiguous()
        return x_

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        frozen_features_list = self.backbone(x)

        if isinstance(frozen_features_list, tuple) or isinstance(frozen_features_list, list):
            frozen_features = torch.stack(frozen_features_list, dim=1)

        # import ipdb; ipdb.set_trace()
        gates = self.learnable_gate(frozen_features) # B, l, k
        frozen_features = torch.einsum("bldhw,blk->bkdhw", frozen_features, gates)
        (b, k, d, h, w) = frozen_features.shape
        frozen_features = frozen_features.reshape(b, k*d, h, w)
        frozen_features = self.frozen_conv(frozen_features) # b, d, h, w

        if self.iters%50==0:
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
            if rank == 0:
                weight_sum = self.learnable_gate.scores.detach().cpu().sum()
                indices = torch.argmax(gates, dim=1) # B, k
                print(f"Self.learnable_gate.gate.sum: {weight_sum}, indices: {indices}")
            self.iters = 0
        self.iters += 1

        ####################
        _, _, H, W = x.shape
        self.h, self.w = math.ceil(H/self.patch_size), math.ceil(W/self.patch_size)
        # split head features
        tuned_features = self.split_head(self.split_activations)
        # import ipdb; ipdb.set_trace()
        tuned_features = self.reshape_vit_tokens(tuned_features)

        x = torch.cat([frozen_features, tuned_features], dim=1)
        x = self.fusion_net(x)

        x1 = self.norm1(self.up1(x))
        x2 = self.norm2(self.up2(x))
        x3 = self.norm3(self.up3(x))
        x4 = self.norm4(self.up4(x))

        return [x1, x2, x3, x4]
