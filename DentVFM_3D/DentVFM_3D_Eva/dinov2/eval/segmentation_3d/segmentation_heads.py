# Author: Tony Xu
#
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from .vit_adapter import ViTAdapter


class UNETRHead(nn.Module):
    def __init__(self, feature_model, input_channels, image_size, num_classes, autocast_ctx, layer_idx=[5, 11, 17, 23], patch_size=[16,16,16]):
        super().__init__()

        self.autocast_ctx = autocast_ctx
        self.input_channels = input_channels
        self.feature_model = feature_model
        self.hidden_size = self.feature_model.num_features
        self.feature_size = 32

        # self.patch_size = self.feature_model.patch_embed.patch_size
        self.patch_size = patch_size
        # self.image_size = image_size
        # self.feat_size = []
        # for i in range(len(self.patch_size)):
        #     self.feat_size.append(self.image_size[i]// self.patch_size[i])
        self.feat_size = [image_size// p for p in self.patch_size]

        # 中间层idx
        self.layer_idx = layer_idx

        # merges multi-channel input into a single vector output for each stage
        self.channel_merge1 = nn.Linear(self.hidden_size * self.input_channels, self.hidden_size)
        self.channel_merge2 = nn.Linear(self.hidden_size * self.input_channels, self.hidden_size)
        self.channel_merge3 = nn.Linear(self.hidden_size * self.input_channels, self.hidden_size)
        self.channel_merge4 = nn.Linear(self.hidden_size * self.input_channels, self.hidden_size)
        self.act_fn = nn.GELU()

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=input_channels,
            out_channels=self.feature_size,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=self.hidden_size,
            out_channels=self.feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=self.hidden_size,
            out_channels=self.feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=self.hidden_size,
            out_channels=self.feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.hidden_size,
            out_channels=self.feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_size * 8,
            out_channels=self.feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_size * 4,
            out_channels=self.feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_size * 2,
            out_channels=self.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=self.feature_size, out_channels=num_classes)
        self.proj_axes = (0, 3 + 1) + tuple(d + 1 for d in range(3))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward_features_multi(self, x_in):
        """Pass multi-channel input through feature model by reshaping batch and merging channels"""
        assert x_in.shape[1] == self.input_channels
        B = x_in.shape[0]

        # Change feature channel into individual batches B, C, H, W, D -> B*C, 1, H, W, D
        x_reshape = x_in.reshape(-1, 1, *x_in.shape[2:])
        with self.autocast_ctx():
            x2, x3, x4, x = self.feature_model.get_intermediate_layers(
                x_reshape,
                n=self.layer_idx,
                return_class_token=False
            )

        # reshape to merge channels B*C, N, F -> B, N, C*F
        x2 = x2.permute(0, 2, 1).reshape(B, self.hidden_size*self.input_channels, -1).permute(0, 2, 1)
        x3 = x3.permute(0, 2, 1).reshape(B, self.hidden_size*self.input_channels, -1).permute(0, 2, 1)
        x4 = x4.permute(0, 2, 1).reshape(B, self.hidden_size*self.input_channels, -1).permute(0, 2, 1)
        x = x.permute(0, 2, 1).reshape(B, self.hidden_size*self.input_channels, -1).permute(0, 2, 1)

        # Merge channels B, N, C*F -> B, N, F
        x2 = self.act_fn(self.channel_merge1(x2))
        x3 = self.act_fn(self.channel_merge2(x3))
        x4 = self.act_fn(self.channel_merge3(x4))
        x = self.act_fn(self.channel_merge4(x))

        return x2, x3, x4, x

    def forward(self, x_in):

        x2, x3, x4, x = self.forward_features_multi(x_in)
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(self.proj_feat(x2))
        enc3 = self.encoder3(self.proj_feat(x3))
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)

# 此处修改
class UNETRHeaforSwim(nn.Module):
    def __init__(self, feature_model, input_channels, image_size, num_classes, autocast_ctx):
        super().__init__()

        self.autocast_ctx = autocast_ctx
        self.input_channels = input_channels
        self.feature_model = feature_model
        self.hidden_size = self.feature_model.num_features
        self.feature_size = 48

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=input_channels,
            out_channels=self.feature_size,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True
        )
        
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.feature_size,
            out_channels=self.feature_size,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=2*self.feature_size,
            out_channels=2*self.feature_size,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=4*self.feature_size,
            out_channels=4*self.feature_size,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True
        )
        
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=16 * self.feature_size,
            out_channels=16 * self.feature_size,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )
        
        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=16 * self.feature_size,
            out_channels=8 * self.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_size * 8,
            out_channels=self.feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_size * 4,
            out_channels=self.feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_size * 2,
            out_channels=self.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_size,
            out_channels=self.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )


        self.out = UnetOutBlock(spatial_dims=3, in_channels=self.feature_size, out_channels=num_classes)
        

    def forward_features_multi(self, x_in):
        """Pass multi-channel input through feature model by reshaping batch and merging channels"""
        assert x_in.shape[1] == self.input_channels
        B = x_in.shape[0]

        # Change feature channel into individual batches B, C, H, W, D -> B*C, 1, H, W, D
        x_reshape = x_in.reshape(-1, 1, *x_in.shape[2:])
        with self.autocast_ctx():
            hidden_states_out = self.feature_model.get_intermediate_layers(
                x_reshape
            )

        return hidden_states_out

    def forward(self, x_in):

        hidden_states_out = self.forward_features_multi(x_in)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        return self.out(out)


class LinearDecoderHead(nn.Module):
    def __init__(self, feature_model, input_channels, image_size, num_classes, autocast_ctx, n_last_layers=4):
        super().__init__()

        self.autocast_ctx = autocast_ctx
        self.input_channels = input_channels

        self.feature_model = feature_model
        self.n_last_layers = n_last_layers
        self.image_size = image_size

        self.hidden_size = self.feature_model.num_features

        # merges multi-channel input into a single vector output
        self.channel_merge = nn.Conv3d(self.hidden_size * self.input_channels, self.hidden_size, kernel_size=1)
        self.act_fn = nn.GELU()

        self.bn_channels = self.hidden_size * n_last_layers
        self.bn = nn.BatchNorm3d(self.bn_channels)
        self.conv_seg = nn.Conv3d(self.bn_channels, num_classes, kernel_size=1)
        self.resize = nn.Upsample(
            size=(self.image_size, ) * 3,
            mode="trilinear"
        )

    def forward_features_multi(self, inputs):
        """Pass multi-channel input through feature model one-by-one and concatenate + merge outputs."""

        assert inputs.shape[1] == self.input_channels
        B = inputs.shape[0]

        # Change feature channel into individual batches B, C, H, W, D -> B*C, 1, H, W, D
        inputs_reshape = inputs.reshape(-1, 1, *inputs.shape[2:])

        with self.autocast_ctx():
            features = self.feature_model.get_intermediate_layers(
                inputs_reshape,
                n=self.n_last_layers,
                return_class_token=False,
                reshape=True
            )

        # reshape to merge channels, B*C, F, H, W, D -> B, C*F, H, W, D
        features = [f.reshape(B, self.hidden_size*self.input_channels, *f.shape[2:]) for f in features]

        # Merge channels B, C*F, H, W, D -> B, C, H, W, D
        merged_feats = [self.act_fn(self.channel_merge(f)) for f in features]
        return merged_feats

    def forward(self, inputs):
        """Forward function."""
        features = self.forward_features_multi(inputs)
        cat_feats = torch.cat(features, dim=1)
        cat_feats = self.bn(cat_feats)
        logits = self.conv_seg(cat_feats)
        return self.resize(logits)


class ViTAdapterUNETRHead(nn.Module):

    def __init__(self, feature_model, input_channels, image_size, num_classes, autocast_ctx):
        super().__init__()

        self.autocast_ctx = autocast_ctx
        self.input_channels = input_channels
        self.feature_model = ViTAdapter(feature_model, input_channels)
        self.hidden_size = self.feature_model.vit_model.num_features
        self.feature_size = 32
        self.patch_size = self.feature_model.vit_model.patch_embed.patch_size
        self.feat_size = [image_size // p for p in self.patch_size]

        self.act_fn = nn.GELU()

        self.encoder1 = UnetrBasicBlock(spatial_dims=3, in_channels=input_channels, out_channels=self.feature_size,
                                        kernel_size=3, stride=1, norm_name='instance', res_block=True)
        self.encoder2 = UnetrBasicBlock(spatial_dims=3, in_channels=self.hidden_size, out_channels=self.feature_size,
                                        kernel_size=3,  stride=1, norm_name='instance', res_block=True)
        self.encoder3 = UnetrBasicBlock(spatial_dims=3, in_channels=self.hidden_size, out_channels=2*self.feature_size,
                                        kernel_size=3, stride=1, norm_name='instance', res_block=True)
        self.encoder4 = UnetrBasicBlock(spatial_dims=3, in_channels=self.hidden_size, out_channels=4*self.feature_size,
                                        kernel_size=3, stride=1, norm_name='instance', res_block=True)
        self.encoder5 = UnetrBasicBlock(spatial_dims=3, in_channels=self.hidden_size, out_channels=8*self.feature_size,
                                        kernel_size=3, stride=1, norm_name='instance', res_block=True)
        self.decoder4 = UnetrUpBlock(spatial_dims=3, in_channels=self.feature_size*8, out_channels=self.feature_size*4,
                                     kernel_size=3, upsample_kernel_size=2, norm_name='instance', res_block=True)
        self.decoder3 = UnetrUpBlock(spatial_dims=3, in_channels=self.feature_size*4, out_channels=self.feature_size*2,
                                     kernel_size=3, upsample_kernel_size=2, norm_name='instance', res_block=True)
        self.decoder2 = UnetrUpBlock(spatial_dims=3, in_channels=self.feature_size*2, out_channels=self.feature_size,
                                     kernel_size=3, upsample_kernel_size=2, norm_name='instance', res_block=True)
        self.decoder1 = UnetrUpBlock(spatial_dims=3, in_channels=self.feature_size, out_channels=self.feature_size,
                                     kernel_size=3, upsample_kernel_size=2, norm_name='instance', res_block=True)
        self.out = UnetOutBlock(spatial_dims=3, in_channels=self.feature_size, out_channels=num_classes)

    def forward(self, x_in):

        f1, f2, f3, f4 = self.feature_model(x_in)
        enc0 = self.encoder1(x_in)  # H, W, D, F
        enc1 = self.encoder2(f1)  # H/2, W/2, D/2, F
        enc2 = self.encoder3(f2)  # H/4, W/4, D/4, 2F
        enc3 = self.encoder4(f3)  # H/8, W/8, D/8, 4F
        enc4 = self.encoder5(f4)  # H/16, W/16, D/16, 8F

        dec2 = self.decoder4(enc4, enc3)  # H/8, W/8, D/8, 4F
        dec1 = self.decoder3(dec2, enc2)  # H/4, W/4, D/4, 2F
        dec0 = self.decoder2(dec1, enc1)  # H/2, W/2, D/2, F
        out = self.decoder1(dec0, enc0)  # H, W, D, F
        return self.out(out)  # H, W, D, num_classes
