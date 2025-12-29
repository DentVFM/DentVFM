from detection_vitsplit import mmcv_custom  # noqa: F401,F403
from detection_vitsplit import mmdet_custom  # noqa: F401,F403
from mmcv import Config, DictAction
from config import resnet_det_config, vitsplit_det_config
from data.dataset import RadiographDataset



def vit_split_config(    
    load_from=None,
    resume_from=None,
    work_dir='',
    fold: int=-1,
    config_file='./detection_vitsplit/configs/mask_rcnn/vitsplit/mask_rcnn_dinov2_vitsplit_base_fpn_1x_coco_ours.py',
    version='',
    affix=''
):
    # cfg = resnet_det_config(load_from, resume_from, work_dir, fold, version)
    cfg = vitsplit_det_config(load_from, resume_from, work_dir, fold, version, affix=affix)

    vitsplit_cfg = Config.fromfile(
        config_file
    )

    cfg.model = vitsplit_cfg.model

    # remove mask head from Mask R-CNN model
    cfg.model.roi_head.pop('mask_head')
    cfg.model.roi_head.pop('mask_roi_extractor')

    # modify number of classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = len(RadiographDataset.CLASSES)

    # cfg.optimizer.lr = 0.00005
    # cfg.optimizer.weight_decay = 0.1
    cfg.optimizer = vitsplit_cfg.optimizer
    cfg.optimizer_config = vitsplit_cfg.optimizer_config
    cfg.lr_config = vitsplit_cfg.lr_config
    cfg.runner.max_epochs = vitsplit_cfg.runner.max_epochs
    cfg.runner.max_epochs = 36
    
    cfg.data.samples_per_gpu = 2
    
    cfg.img_norm_cfg.mean = [123.675, 116.28, 103.53]
    cfg.img_norm_cfg.std = [58.395, 57.12, 57.375]
    cfg.data.train.pipeline[4].mean = [123.675, 116.28, 103.53]
    cfg.data.train.pipeline[4].std = [58.395, 57.12, 57.375]
    cfg.train_pipeline[4].mean = [123.675, 116.28, 103.53]
    cfg.train_pipeline[4].std = [58.395, 57.12, 57.375]
    cfg.data.val.pipeline[1].transforms[2].mean = [123.675, 116.28, 103.53]
    cfg.data.val.pipeline[1].transforms[2].std = [58.395, 57.12, 57.375]
    cfg.data.test.pipeline[1].transforms[2].mean = [123.675, 116.28, 103.53]
    cfg.data.test.pipeline[1].transforms[2].std = [58.395, 57.12, 57.375]
    cfg.test_pipeline[1].transforms[2].mean = [123.675, 116.28, 103.53]
    cfg.test_pipeline[1].transforms[2].std = [58.395, 57.12, 57.375]

    # 新增配置
    # cfg.optimizer.paramwise_cfg.layer_decay_rate = 0.9
    # cfg.optimizer.lr = 0.00005

    return cfg

def vit_split_config_ori(    
    load_from=None,
    resume_from=None,
    work_dir='checkpoints1',
    fold: int=-1,
):
    cfg = resnet_det_config(load_from, resume_from, work_dir, fold)

    vitsplit_cfg = Config.fromfile(
        './detection_vitsplit/configs/mask_rcnn/vitsplit/mask_rcnn_dinov2_vitsplit_base_fpn_1x_coco.py',
    )

    cfg.model = vitsplit_cfg.model

    # remove mask head from Mask R-CNN model
    cfg.model.roi_head.pop('mask_head')
    cfg.model.roi_head.pop('mask_roi_extractor')

    # modify number of classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = len(RadiographDataset.CLASSES)

    # cfg.optimizer.lr = 0.00005
    # cfg.optimizer.weight_decay = 0.1
    cfg.optimizer = vitsplit_cfg.optimizer
    cfg.optimizer_config = vitsplit_cfg.optimizer_config
    cfg.lr_config = vitsplit_cfg.lr_config
    cfg.runner.max_epochs = vitsplit_cfg.runner.max_epochs
    
    cfg.data.samples_per_gpu = 1
    
    cfg.img_norm_cfg.mean = [123.675, 116.28, 103.53]
    cfg.img_norm_cfg.std = [58.395, 57.12, 57.375]
    cfg.data.train.pipeline[4].mean = [123.675, 116.28, 103.53]
    cfg.data.train.pipeline[4].std = [58.395, 57.12, 57.375]
    cfg.train_pipeline[4].mean = [123.675, 116.28, 103.53]
    cfg.train_pipeline[4].std = [58.395, 57.12, 57.375]
    cfg.data.val.pipeline[1].transforms[2].mean = [123.675, 116.28, 103.53]
    cfg.data.val.pipeline[1].transforms[2].std = [58.395, 57.12, 57.375]
    cfg.data.test.pipeline[1].transforms[2].mean = [123.675, 116.28, 103.53]
    cfg.data.test.pipeline[1].transforms[2].std = [58.395, 57.12, 57.375]
    cfg.test_pipeline[1].transforms[2].mean = [123.675, 116.28, 103.53]
    cfg.test_pipeline[1].transforms[2].std = [58.395, 57.12, 57.375]
    return cfg