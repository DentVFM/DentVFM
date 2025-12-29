from pathlib import Path
import pickle

import torch


path = Path('checkpoints/model_final_e5f7ce.pkl')
with open(path, 'rb') as f:
    d2_ckpt = pickle.load(f)['model']

mmdet_ckpt = {}
for key, value in d2_ckpt.items():
    key_parts = key.split('.')
    if 'roi_heads.box_head' in key:
        layer = int(key_parts[2][-1]) - 1
        key = f'roi_head.bbox_head.shared_fcs.{layer}.{key_parts[-1]}'
    elif 'roi_heads.box_predictor.cls_score' in key:
        key = f'roi_head.bbox_head.fc_cls.{key_parts[-1]}'
    elif 'roi_heads.box_predictor.bbox_pred' in key:
        key = f'roi_head.bbox_head.fc_reg.{key_parts[-1]}'
    elif 'stem' in key:
        if 'norm' in key:
            key = f'backbone.bn1.{key_parts[-1]}'
        else:
            key = f'backbone.conv1.{key_parts[-1]}'
    elif 'bottom_up' in key:
        block = int(key_parts[2][-1]) - 1
        if 'shortcut' in key:
            if 'norm' in key:
                key = f'backbone.layer{block}.0.downsample.1.{key_parts[-1]}'
            else:
                key = f'backbone.layer{block}.0.downsample.0.{key_parts[-1]}'
        else:
            layer = int(key_parts[4][-1])
            if 'norm' in key:
                key = f'backbone.layer{block}.{key_parts[3]}.bn{layer}.{key_parts[-1]}'
            else:
                key = f'backbone.layer{block}.{key_parts[3]}.{key_parts[-2]}.{key_parts[-1]}'
    elif 'fpn_lateral' in key:
        layer = int(key_parts[1][-1]) - 2
        key = f'neck.lateral_convs.{layer}.conv.{key_parts[-1]}'
    elif 'fpn_output' in key:
        layer = int(key_parts[1][-1]) - 2
        key = f'neck.fpn_convs.{layer}.conv.{key_parts[-1]}'
    else:
        print(key)

    mmdet_ckpt[key] = torch.from_numpy(value)



with open('checkpoints/fast_rcnn_mmdet.pth', 'wb') as f:
    ckpt = {'state_dict': mmdet_ckpt}
    torch.save(ckpt, f)

k = 3
