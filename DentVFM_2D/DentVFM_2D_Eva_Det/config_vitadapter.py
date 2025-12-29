from detection_vitadapter import mmcv_custom  # noqa: F401,F403
from detection_vitadapter import mmdet_custom  # noqa: F401,F403
from mmcv import Config, DictAction
from config import resnet_det_config, vitadapter_det_config
from data.dataset import RadiographDataset



def vit_adapter_config(    
    load_from=None,
    resume_from=None,
    work_dir='./checkpoints',
    fold: int=-1,
    config_file='./detection_vitadapter/configs/mask_rcnn/dinov2/mask_rcnn_dinov2_adapter_base_fpn_3x_coco_ours_v1.py',
    version='',
    affix=''
):
    # cfg = resnet_det_config(load_from, resume_from, work_dir, fold)
    cfg = vitadapter_det_config(load_from, resume_from, work_dir, fold, version, affix=affix)

    vitadapter_cfg = Config.fromfile(
        config_file
        # './detection_vitadapter/configs/mask_rcnn/dinov2/mask_rcnn_dinov2_adapter_base_fpn_3x_coco_ours_v1.py',
        # './detection_vitadapter/configs/mask_rcnn/dinov2/mask_rcnn_dinov2_adapter_base_fpn_3x_coco_ours_v2.py',
    )

    cfg.model = vitadapter_cfg.model

    # cfg = vitadapter_det_config(load_from, resume_from, work_dir, fold)

    # remove mask head from Mask R-CNN model
    cfg.model.roi_head.pop('mask_head')
    cfg.model.roi_head.pop('mask_roi_extractor')

    # modify number of classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = len(RadiographDataset.CLASSES)

    # cfg.optimizer.lr = 0.00005
    # cfg.optimizer.weight_decay = 0.1
    cfg.optimizer = vitadapter_cfg.optimizer
    cfg.optimizer_config = vitadapter_cfg.optimizer_config
    cfg.lr_config = vitadapter_cfg.lr_config
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
    # cfg.model.rpn_head.anchor_generator = dict(
    #         type='AnchorGenerator',
    #         ratios=[0.5, 1.0, 2.0],
    #         strides=[4, 8, 16, 32, 64],
    #         octave_base_scale=8,
    #         scales_per_octave=3
    #     )
    # cfg.model.rpn_head.bbox_coder = dict(
    #         type='DeltaXYWHBBoxCoder',
    #         target_means=[0., 0., 0., 0.],
    #         target_stds=[0.1, 0.1, 0.2, 0.2]   # 关键修正
    #     )
    # cfg.model.rpn_head.loss_bbox = dict(type='SmoothL1Loss', beta=1.0/9.0, loss_weight=1.0)
    # cfg.model.roi_head.bbox_head.loss_bbox = dict(type='SmoothL1Loss', beta=1.0/9.0, loss_weight=1.0)
    # cfg.model.train_cfg.rpn.assigner.pos_iou_thr = 0.5 
    # cfg.model.train_cfg.rcnn.sampler.pos_fraction = 0.33
    cfg.optimizer.paramwise_cfg.layer_decay_rate = 0.9
    # cfg.data.samples_per_gpu = 4
    # cfg.optimizer.lr = 0.00005
    
    return cfg



def vit_adapter_config_ori(    
    load_from='',
    resume_from=None,
    work_dir='checkpoints',
    fold: int=-1,
):
    cfg = resnet_det_config(load_from, resume_from, work_dir, fold)

    vitadapter_cfg = Config.fromfile(
        'detection_vitadapter/configs/mask_rcnn/dinov2/mask_rcnn_dinov2_adapter_base_fpn_3x_coco.py'
    )

    cfg.model = vitadapter_cfg.model

    # remove mask head from Mask R-CNN model
    cfg.model.roi_head.pop('mask_head')
    cfg.model.roi_head.pop('mask_roi_extractor')

    # modify number of classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = len(RadiographDataset.CLASSES)

    # cfg.optimizer.lr = 0.00005
    # cfg.optimizer.weight_decay = 0.1
    cfg.optimizer = vitadapter_cfg.optimizer
    cfg.optimizer_config = vitadapter_cfg.optimizer_config
    cfg.lr_config = vitadapter_cfg.lr_config
    cfg.runner.max_epochs = 36
    
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