import torch.nn as nn
import torch
from typing import Sequence, Union, Tuple
import math
from .vision_transformer import DinoVisionTransformer
from .load_dino_weights import dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14, dinov2_vits14, \
    dinov2_vits14_reg, dinov2_vitb14_reg, dinov2_vitl14_reg, dinov2_vitg14_reg, dinov2_vitb14_dentvfm

def interpolate_pos_encoding(self, x, w, h):
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return self.pos_embed
    pos_embed = self.pos_embed.float()
    # class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // self.patch_size
    h0 = h // self.patch_size
    M = int(math.sqrt(N))  # Recover the number of patches in each dimension
    assert N == M * M
    kwargs = {}
    if self.interpolate_offset:
        # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
        # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
        sx = float(w0 + self.interpolate_offset) / M
        sy = float(h0 + self.interpolate_offset) / M
        kwargs["scale_factor"] = (sx, sy)
    else:
        # Simply specify an output size instead of a scale factor
        kwargs["size"] = (w0, h0)
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
        mode="bicubic",
        # antialias=self.interpolate_antialias,
        **kwargs,
    )
    assert (w0, h0) == patch_pos_embed.shape[-2:]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    # return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
    return patch_pos_embed.to(previous_dtype)

def prepare_tokens_with_masks(self, x, masks=None):
    B, nc, w, h = x.shape
    x = self.patch_embed(x)
    if masks is not None:
        x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

    # x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + self.interpolate_pos_encoding(x, w, h)
    # x = x # remove pos embedding

    if self.register_tokens is not None:
        x = torch.cat(
            (
                # x[:, 0], # cls token
                self.register_tokens.expand(x.shape[0], -1, -1),
                x[:, 1:],
            ),
            dim=1,
        )

    return x

def get_intermediate_layers(
    self,
    x: torch.Tensor,
    n: Union[int, Sequence] = 1,  # Layers or n last layers to take
    reshape: bool = False,
    return_class_token: bool = False,
    norm=True,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
    if self.chunked_blocks:
        outputs = self._get_intermediate_layers_chunked(x, n)
    else:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
    if norm:
        outputs = [self.norm(out) for out in outputs]
    # class_tokens = [out[:, 0] for out in outputs]
    outputs = [out[:, self.num_register_tokens :] for out in outputs]
    if reshape:
        B, _, w, h = x.shape
        outputs = [
            out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
            for out in outputs
        ]
    # if return_class_token:
    #     return tuple(zip(outputs, class_tokens))
    return tuple(outputs)

def hack_dinov2():
    """
    remove cls token
    """
    DinoVisionTransformer.interpolate_pos_encoding = interpolate_pos_encoding
    DinoVisionTransformer.prepare_tokens_with_masks = prepare_tokens_with_masks
    DinoVisionTransformer.get_intermediate_layers = get_intermediate_layers

def initialize_model(backbone_name=None, **kwargs):
    # hack_dinov2() 
    # import ipdb; ipdb.set_trace()
    if backbone_name=='vits14':
        model = dinov2_vits14(**kwargs)
    elif backbone_name=='vitb14':
        model = dinov2_vitb14(**kwargs)
    elif backbone_name=='vitl14':
        model = dinov2_vitl14(**kwargs)
    elif backbone_name=='vitg14':
        model = dinov2_vitg14(**kwargs)
    elif backbone_name=='vitb14_dentvfm':
        model = dinov2_vitb14_dentvfm(**kwargs)
    else:
        raise TypeError("This type of model doesn't exist !!!")

    return model
