# Author: Tony Xu
#
# This code is adapted from the original ViT-Adapter repository: https://github.com/czczup/ViT-Adapter
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs, MSDeformAttn, InteractionBlockWithCls


class ViTAdapter(nn.Module):
    def __init__(self, vit_model, input_channels, pretrain_size=112, conv_inplane=32, n_points=4,
                 deform_num_heads=8, init_values=1e-6, with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=0.25, add_vit_feature=True,
                 use_extra_extractor=True, with_cp=True, use_cls=True, drop_path_rate=0.2):

        super().__init__()
        self.vit_model = vit_model
        self.use_cls = use_cls
        self.drop_path_rate = drop_path_rate
        self.input_channels = input_channels

        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

        n_indexes = 4
        self.indexes = [[0, 5], [6, 11], [12, 17], [18, 23]]

        self.num_block = len(self.vit_model.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size, pretrain_size)
        self.add_vit_feature = add_vit_feature
        embed_dim = self.vit_model.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(
            inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False, in_channels=input_channels)
        block_fn = InteractionBlockWithCls if use_cls else InteractionBlock
        self.interactions = nn.Sequential(*[
            block_fn(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                     init_values=init_values, drop_path=self.drop_path_rate,
                     norm_layer=self.norm_layer, with_cffn=with_cffn,
                     cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                     extra_extractor=((True if i == n_indexes - 1
                                       else False) and use_extra_extractor),
                     with_cp=with_cp)
            for i in range(n_indexes)
        ])
        self.up = nn.ConvTranspose3d(embed_dim, embed_dim, 2, 2)
        self.channel_merge1 = nn.Conv3d(embed_dim * self.input_channels, embed_dim, kernel_size=1)
        self.channel_merge2 = nn.Conv3d(embed_dim * self.input_channels, embed_dim, kernel_size=1)
        self.channel_merge3 = nn.Conv3d(embed_dim * self.input_channels, embed_dim, kernel_size=1)
        self.channel_merge4 = nn.Conv3d(embed_dim * self.input_channels, embed_dim, kernel_size=1)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W, D):
        pos_embed = pos_embed.reshape(
            1,
            self.pretrain_size[0] // 16,
            self.pretrain_size[1] // 16,
            self.pretrain_size[2] // 16,
            -1
        ).permute(0, 4, 1, 2, 3)
        pos_embed = F.interpolate(pos_embed, size=(H, W, D), mode='trilinear', align_corners=False).reshape(
            1, -1, H * W * D).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Convert to 1 channel input for pretrained ViT
        bs, H, W, D = x.shape[0], x.shape[2] // 16, x.shape[3] // 16, x.shape[4] // 16
        x = x.reshape(bs * self.input_channels, 1, *x.shape[2:])
        # Patch Embedding forward
        x = self.vit_model.patch_embed(x)
        _, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.vit_model.pos_embed[:, 1:], H, W, D)

        # use cls token
        if self.use_cls:
            cls_token = self.vit_model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            pos_embed = torch.cat((self.vit_model.pos_embed[:, :1], pos_embed), dim=1)

        x = x + pos_embed

        # Interaction
        # Computes each channel individually in ViT, merges channels by averaging for injector and extractor
        if self.use_cls:
            cls, x = (x[:, :1], x[:, 1:])
        outs = list()
        for i, layer in enumerate(self.interactions):
            if self.vit_model.chunked_blocks:
                if self.use_cls:
                    x, c, cls = layer(x, c, cls, self.vit_model.blocks[i], deform_inputs1, deform_inputs2,
                                      H, W, D, self.input_channels)
                else:
                    x, c = layer(x, c, self.vit_model.blocks[i], deform_inputs1, deform_inputs2,
                                 H, W, D, self.input_channels)
            else:
                st, ed = self.indexes[i]
                if self.use_cls:
                    x, c, cls = layer(x, c, cls, self.vit_model.blocks[st: ed + 1], deform_inputs1, deform_inputs2,
                                      H, W, D, self.input_channels)
                else:
                    x, c = layer(x, c, self.vit_model.blocks[st: ed + 1], deform_inputs1, deform_inputs2,
                                 H, W, D, self.input_channels)
            outs.append(x.transpose(1, 2).view(bs * self.input_channels, dim, H, W, D).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 4, W * 4, D * 4).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H * 2, W * 2, D * 2).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H, W, D).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = self.channel_merge1(x1.view(bs, self.input_channels * dim, H, W, D))
            x2 = self.channel_merge2(x2.view(bs, self.input_channels * dim, H, W, D))
            x3 = self.channel_merge3(x3.view(bs, self.input_channels * dim, H, W, D))
            x4 = self.channel_merge4(x4.view(bs, self.input_channels * dim, H, W, D))

            x1 = F.interpolate(x1, scale_factor=8, mode='trilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=4, mode='trilinear', align_corners=False)
            x3 = F.interpolate(x3, scale_factor=2, mode='trilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]
