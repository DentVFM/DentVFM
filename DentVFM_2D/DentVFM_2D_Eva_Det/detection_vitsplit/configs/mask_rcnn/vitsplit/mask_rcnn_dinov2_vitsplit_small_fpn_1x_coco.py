_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/coco_instance.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

#############################################
initial_lr = 1e-4
wd = 5e-2
layer_decay_rate = 0.76
vit_hidden_dims = 384
pergpu_batch_size = 2
frozen_num = 5
tuned_num = 11
start = 2
gap = (11-start)/(frozen_num-1)
frozen_indices = [start+int(idx*gap) for idx in range(frozen_num)]
tuned_indices = [idx for idx in range(12-tuned_num, 12)]
register_version = False
tune_register = False
dir_name = f"maskrcnn_fpn1x_dinov2_vits14_coco_splithead_{frozen_num}frozen_layers_start{start}_{tuned_num}tuning_layers_lr1e-4_lrlayerdecay{layer_decay_rate}_4pyramids_fpnneck_256dim_seed202503"
#############################################

# Add this to ensure unused parameters are handled
find_unused_parameters = True

model = dict(
    backbone=dict(
                  type="DINOViTSplitFusion", 
                  backbone_size = "small",
                  register_version = register_version,
                  tune_register = tune_register,
                  out_indices=frozen_indices,
                  select_layers=tuned_indices,
                  channels=vit_hidden_dims,
                  tuning_type = "frozen"
                ),
    neck=dict(
        type='FPN',
        in_channels=[vit_hidden_dims, vit_hidden_dims, vit_hidden_dims, vit_hidden_dims],
        out_channels=256,
        num_outs=5),
    )
# optimizer
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='RandomCrop',
         crop_type='absolute_range',
         crop_size=(1024, 1024),
         allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(
    samples_per_gpu=pergpu_batch_size,
    train=dict(pipeline=train_pipeline))
optimizer = dict(
    _delete_=True, type='AdamW', lr=initial_lr, weight_decay=wd,
    constructor='LayerDecayOptimizerConstructorSplit',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=layer_decay_rate)
    )
optimizer_config = dict(grad_clip=None)
log_config = dict(interval=50, hooks=[dict(type="PrintLrGroupHook", by_epoch=True)])
# lr_config = dict(
#     policy="poly",
#     warmup="linear",
#     warmup_iters=1500,
#     warmup_ratio=1e-06,
#     power=0.9,
#     min_lr=0.0,
#     by_epoch=False,
# )
fp16 = dict(loss_scale=dict(init_scale=512))
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,
    save_last=True,
)
work_dir = f"./work_dirs/{dir_name}"
