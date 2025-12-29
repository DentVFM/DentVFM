import torch

if not hasattr(torch._tensor, "_rebuild_from_type_v2"):

    def _rebuild_from_type_v2(func, new_type, args, state, *, archiveinfo=None):
        t = func(*args)

        try:
            t.__class__ = new_type
        except Exception:
            pass

        if isinstance(state, dict):
            for k, v in state.items():
                setattr(t, k, v)

        elif isinstance(state, tuple) and len(state) == 2:
            dict_state, slots = state

            if isinstance(dict_state, dict):
                for k, v in dict_state.items():
                    setattr(t, k, v)

            if slots is not None:
                for k, v in slots.items():
                    setattr(t, k, v)
        return t

    torch._tensor._rebuild_from_type_v2 = _rebuild_from_type_v2

try:
    import torch.distributed as dist

    if dist.is_available():
        _orig_get_world_size = dist.get_world_size

        def get_world_size_safe(group=None):
            if not dist.is_initialized():
                return 1
            try:
                return _orig_get_world_size(group)
            except RuntimeError:
                return 1

        dist.get_world_size = get_world_size_safe
except Exception:
    pass

import argparse
import os
import mmcv
import warnings
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, single_gpu_test
from mmdet.utils import build_dp, build_ddp
from mmcv.runner import get_dist_info, init_dist
from config import init_config, resnet_det_config, fast_det_config, rpn_config
from data.dataset import RadiographDataset


def train(args):
    # mask rcnn swint
    # cfg = init_config(fold=args.fold)

    if args.adaptername == 'swint':
        cfg = init_config(fold=args.fold)

    # vit adapter
    if args.adaptername == 'vitadapter':
        from config_vitadapter import vit_adapter_config, vit_adapter_config_ori
        cfg = vit_adapter_config(fold=args.fold,
                                    work_dir='',
                                    config_file='./detection_vitadapter/configs/mask_rcnn/dinov2/mask_rcnn_dinov2_adapter_base_fpn_3x_coco_ours_v1.py',
                                    version='',
                                    affix='')
    
    if args.adaptername == 'vitsplit':
        # vit split
        from config_vitsplit import vit_split_config, vit_split_config_ori
        if args.modelname == 'dentvfm':
            cfg = vit_split_config(fold=args.fold,
                                work_dir='',
                                config_file='./detection_vitsplit/configs/mask_rcnn/vitsplit/mask_rcnn_dinov2_vitsplit_base_fpn_1x_coco_ours.py',
                                version='',
                                affix=''
                                )

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute training time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
        
    cfg.device = 'cuda' # fix 'ConfigDict' object has no attribute 'device'

    # build dataset
    datasets = [build_dataset(cfg.data.train)]

    # build detector
    model = build_detector(cfg.model)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    # add an attribute for visualization convenience
    model.CLASSES = RadiographDataset.CLASSES
    
    # model = build_dp(model, device=cfg.device, device_ids=cfg.gpu_ids)
    # model = build_ddp(model, device=cfg.device)

    # train detector
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        meta={'config': cfg.pretty_text, 'seed': 0},
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--fold', type=int, default=0)
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus',
                            type=int,
                            help='number of gpus to use '
                            '(only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids',
                            type=int,
                            nargs='+',
                            help='ids of gpus to use '
                            '(only applicable to non-distributed training)')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--modelname', type=str, default='dentvfm')
    # parser.add_argument('--adaptername', type=str, default='vitadapter')
    parser.add_argument('--adaptername', type=str, default='vitsplit')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    train(args)