# Modified from https://github.com/open-mmlab/mmsegmentation/blob/main/tools/train.py

# Copyright (c) OpenMMLab. All rights reserved.
# License: Apache License 2.0 https://github.com/open-mmlab/mmsegmentation/blob/main/LICENSE

import sys
sys.path.append("/DentVFM/DentVFM_2D/DentVFM_2D_Eva_Seg")
import os
import argparse
import logging
import os
import os.path as osp
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmseg.registry import RUNNERS
from segmentation_m2f import *
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config',
                        default='./models/mask2former_swin-t_8xb2-90k_cityscapes-512x1024.py',
                        help='train config file path')
    parser.add_argument('--work-dir',
                        default='./workdir',
                        help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    
    parser.add_argument(
        "--save_exist",
        action="store_true",
        default=False,
        help="do not remove existing dir",
    )
    parser.add_argument(
        "--data_root",
        default='Children_dental_caries_segmentation_v1_fold1'
    )
    parser.add_argument(
        "--pretrained"
    )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.data_root = os.path.join(os.path.dirname(cfg.data_root), args.data_root)
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.test_dataloader = cfg.val_dataloader

    if args.pretrained == "keep_origin":
        if 'backbone_ckp' not in cfg:
            print("without update")
        else:
            print("current model:", cfg.backbone_ckp, "without update")
    else:
        print("current model:", args.backbone_ckp, "will be updated to config file!")
        cfg.backbone_ckp = args.pretrained
        cfg.model.backbone.pretrained = args.pretrained

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir

        if osp.exists(cfg.work_dir) and not args.save_exist:
            print("work_dir exist!! delete...")
            for item in os.listdir(cfg.work_dir):
                item_path = os.path.join(cfg.work_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            print("work_dir clean !")
        else:
            if osp.exists(cfg.work_dir):
                print("work_dir exist!! not delete")
            else:
                print("work_dir don't exist!! will create")

    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()