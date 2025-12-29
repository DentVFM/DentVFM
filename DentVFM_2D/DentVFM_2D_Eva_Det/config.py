import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed

from data.dataset import RadiographDataset


# mask rcnn swimt
def init_config(
    load_from='',
    resume_from=None,
    work_dir='',
    fold: int=-1,
):
    cfg = resnet_det_config(load_from, resume_from, work_dir, fold)
    
    cfg_prefix = 'mmdetection/configs/'
    swin_cfg = Config.fromfile(
        cfg_prefix + 'swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py',
    )

    cfg.model = swin_cfg.model

    # remove mask head from Mask R-CNN model
    cfg.model.roi_head.pop('mask_head')
    cfg.model.roi_head.pop('mask_roi_extractor')

    # modify number of classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = len(RadiographDataset.CLASSES)

    cfg.optimizer = swin_cfg.optimizer
    cfg.optimizer.lr = 0.00005
    cfg.optimizer.weight_decay = 0.1

    return cfg


def resnet_det_config(
    load_from='',
    resume_from=None,
    work_dir='result',
    fold: int=-1,        
):
    # load Faster R-CNN configuration with swin model
    cfg_prefix = 'mmdetection/configs/'
    cfg = Config.fromfile(
        cfg_prefix + 'faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py',
    )
    
    fold = '' if fold == -1 else str(fold)

    ############################################################################
    # Data settings                                                            #
    ############################################################################
    # modify general dataset settings
    cfg.dataset_type = 'RadiographDataset'
    cfg.data_root = 'XXXX'
    cfg.img_norm_cfg.mean = [136.88, 136.88, 136.88]
    cfg.img_norm_cfg.std = [61.59, 61.59, 61.59]

    # modify train set settings
    cfg.data.train.type = 'RadiographDataset'
    cfg.data.train.data_root = 'XXXX'
    cfg.data.train.ann_file = f'traini{fold}.txt'
    cfg.data.train.img_prefix = 'radiographs'
    cfg.data.train.pipeline[4].mean = [136.88, 136.88, 136.88]
    cfg.data.train.pipeline[4].std = [61.59, 61.59, 61.59]
    cfg.train_pipeline[4].mean = [136.88, 136.88, 136.88]
    cfg.train_pipeline[4].std = [61.59, 61.59, 61.59]

    # modify validation set settings
    cfg.data.val.type = 'RadiographDataset'
    cfg.data.val.data_root = 'XXXX'
    cfg.data.val.ann_file = f'vali{fold}.txt'
    cfg.data.val.img_prefix = 'radiographs'
    cfg.data.val.pipeline[1].transforms[2].mean = [136.88, 136.88, 136.88]
    cfg.data.val.pipeline[1].transforms[2].std = [61.59, 61.59, 61.59]

    # modify test set settings
    cfg.data.test.type = 'RadiographDataset'
    cfg.data.test.data_root = 'XXXX'
    cfg.data.test.ann_file = f'testi{fold}.txt'
    cfg.data.test.img_prefix = 'radiographs'
    cfg.data.test.pipeline[1].transforms[2].mean = [136.88, 136.88, 136.88]
    cfg.data.test.pipeline[1].transforms[2].std = [61.59, 61.59, 61.59]
    cfg.data.test.pipeline[1].img_scale = [
        (1333, 640), (1333, 672), (1333, 704),
        (1333, 736), (1333, 768), (1333, 800),
    ]
    cfg.data.test.pipeline[1].flip = True
    cfg.test_pipeline[1].transforms[2].mean = [136.88, 136.88, 136.88]
    cfg.test_pipeline[1].transforms[2].std = [61.59, 61.59, 61.59]
    cfg.test_pipeline[1].img_scale = [
        (1333, 640), (1333, 672), (1333, 704),
        (1333, 736), (1333, 768), (1333, 800),
    ]
    cfg.test_pipeline[1].flip = True

    ############################################################################
    # Model settings                                                           #
    ############################################################################
    # modify number of classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = len(RadiographDataset.CLASSES)

    # load model from checkpoint to either start or resume training with
    cfg.load_from = load_from
    cfg.resume_from = resume_from


    ############################################################################
    # Training settings                                                        #
    ############################################################################
    # train for 12 epochs, dividing LR by 10 afer epochs 8 and 11
    cfg.checkpoint_config = None

    cfg.optimizer.lr = 0.00005
    cfg.optimizer.weight_decay = 0.1
    cfg.lr_config.warmup = None
    cfg.lr_config.step = [12, 8, 11]
    cfg.runner.max_epochs = 12

    cfg.data.samples_per_gpu = 2

    # cfg.lr_config.step = [5, 7]
    # cfg.runner.max_epochs = 8

    cfg.lr_config.step = [16, 22]
    cfg.runner.max_epochs = 24

    # only one GPU is used
    cfg.device = 'cuda'
    cfg.gpu_ids = range(1)

    ############################################################################
    # Miscellaneous                                                            #
    ############################################################################    
    # set up working dir to save checkpoint and logs.
    # cfg.work_dir = work_dir + f'/{fold}_resnet'
    cfg.work_dir = work_dir + f'/fold_{fold}'
    mmcv.mkdir_or_exist(cfg.work_dir)

    # show train results every 10 batches
    cfg.log_config.interval = 10

    # show validation mAP results and save checkpoint every 1 epoch
    cfg.evaluation.metric = 'mAP'
    cfg.evaluation.interval = 1
    cfg.evaluation.save_best = 'mAP'
    # cfg.checkpoint_config.interval = 1

    cfg.mp_start_method = 'spawn'

    # set seed so the results are reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)

    # initialize logger and show final configuration
    print(f'Config:\n{cfg.pretty_text}')

    return cfg


def fast_det_config(
    load_from='checkpoints/fast_rcnn_mmdet.pth',
    resume_from=None,
    work_dir='',
    fold: int=-1,
):
    # load Faster R-CNN configuration with swin model
    cfg_prefix = 'mmdetection/configs/'
    cfg = Config.fromfile(
        cfg_prefix + 'fast_rcnn/fast_rcnn_r50_fpn_2x_coco.py',
    )

    print(cfg.pretty_text)

    fold = '' if fold == -1 else str(fold)  

    # set up working dir to save checkpoint and logs.
    cfg.work_dir = work_dir + f'/{fold}_fast'
    mmcv.mkdir_or_exist(cfg.work_dir)

    ############################################################################
    # Data settings                                                            #
    ############################################################################
    # modify general dataset settings
    cfg.dataset_type = 'RadiographDataset'
    cfg.data_root = 'XXXX'
    cfg.img_norm_cfg.mean = [136.88, 136.88, 136.88]
    cfg.img_norm_cfg.std = [61.59, 61.59, 61.59]

    # modify train set settings
    cfg.data.train.type = 'RadiographDataset'
    cfg.data.train.data_root = 'XXXX'
    cfg.data.train.ann_file = f'Osteolysis_Carotis/traini{fold}.txt'
    cfg.data.train.proposal_file = f'result/{fold}_rpn/results_train.pkl'
    cfg.data.train.img_prefix = 'Osteolysis_Carotis/radiographs'
    cfg.data.train.pipeline[5].mean = [136.88, 136.88, 136.88]
    cfg.data.train.pipeline[5].std = [61.59, 61.59, 61.59]
    cfg.train_pipeline[5].mean = [136.88, 136.88, 136.88]
    cfg.train_pipeline[5].std = [61.59, 61.59, 61.59]

    # modify validation set settings
    cfg.data.val.type = 'RadiographDataset'
    cfg.data.val.data_root = 'XXXX'
    cfg.data.val.ann_file = f'Osteolysis_Carotis/vali{fold}.txt'
    cfg.data.val.proposal_file = f'result/{fold}_rpn/results_val.pkl'
    cfg.data.val.img_prefix = 'Osteolysis_Carotis/radiographs'
    cfg.data.val.pipeline[2].transforms[2].mean = [136.88, 136.88, 136.88]
    cfg.data.val.pipeline[2].transforms[2].std = [61.59, 61.59, 61.59]

    # modify test set settings
    cfg.data.test.type = 'RadiographDataset'
    cfg.data.test.data_root = 'XXXX'
    cfg.data.test.ann_file = f'Osteolysis_Carotis/test{fold}.txt'
    cfg.data.test.proposal_file = f'result/{fold}_rpn/results_test.pkl'
    cfg.data.test.img_prefix = 'Osteolysis_Carotis/radiographs'
    cfg.data.test.pipeline[2].transforms[2].mean = [136.88, 136.88, 136.88]
    cfg.data.test.pipeline[2].transforms[2].std = [61.59, 61.59, 61.59]
    cfg.data.test.pipeline[2].img_scale = [(1333, 800)
        # (1333, 640), (1333, 672), (1333, 704),
        # (1333, 736), (1333, 768), (1333, 800),
    ]
    cfg.data.test.pipeline[2].flip = False
    cfg.test_pipeline[2].transforms[2].mean = [136.88, 136.88, 136.88]
    cfg.test_pipeline[2].transforms[2].std = [61.59, 61.59, 61.59]
    cfg.test_pipeline[2].img_scale = [(1333, 800)
        # (1333, 640), (1333, 672), (1333, 704),
        # (1333, 736), (1333, 768), (1333, 800),
    ]
    cfg.test_pipeline[2].flip = False


    ############################################################################
    # Model settings                                                           #
    ############################################################################
    # modify number of classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = len(RadiographDataset.CLASSES)

    # load model from checkpoint to either start or resume training with
    cfg.load_from = load_from
    cfg.resume_from = resume_from


    ############################################################################
    # Training settings                                                        #
    ############################################################################
    # train for 12 epochs, dividing LR by 10 afer epochs 8 and 11
    # cfg.checkpoint_config = None

    cfg.optimizer.lr = 0.00005
    cfg.optimizer.weight_decay = 0.1
    cfg.lr_config.warmup = None
    cfg.lr_config.step = [12, 8, 11]
    cfg.runner.max_epochs = 12

    cfg.data.samples_per_gpu = 2


    # cfg.lr_config.step = [5, 7]
    # cfg.runner.max_epochs = 8


    cfg.lr_config.step = [16, 22]
    cfg.runner.max_epochs = 24

    # only one GPU is used
    cfg.device = 'cuda'
    cfg.gpu_ids = range(1)


    ############################################################################
    # Miscellaneous                                                            #
    ############################################################################  

    # show train results every 10 batches
    cfg.log_config.interval = 10

    # show validation mAP results and save checkpoint every 1 epoch
    cfg.evaluation.metric = 'mAP'
    cfg.evaluation.interval = 1
    cfg.evaluation.save_best = 'mAP'
    cfg.checkpoint_config.interval = 1
    cfg.checkpoint_config.max_keep_ckpts = 1


    cfg.mp_start_method = 'spawn'

    # set seed so the results are reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)

    # initialize logger and show final configuration
    print(f'Config:\n{cfg.pretty_text}')

    return cfg


def rpn_config(
    load_from='checkpoints/rpn_r50.pth',
    work_dir='',
    fold=-1,
    proposals='test',
):
    # load Faster R-CNN configuration with swin model
    cfg_prefix = 'mmdetection/configs/'
    cfg = Config.fromfile(
        cfg_prefix + 'rpn/rpn_r50_fpn_2x_coco.py',
    )



    fold = '' if fold == -1 else str(fold)

    ############################################################################
    # Data settings                                                            #
    ############################################################################
    # modify general dataset settings
    cfg.dataset_type = 'RadiographDataset'
    cfg.data_root = 'XXXX'
    cfg.img_norm_cfg.mean = [136.88, 136.88, 136.88]
    cfg.img_norm_cfg.std = [61.59, 61.59, 61.59]

    # modify train set settings
    cfg.data.train.type = 'RadiographDataset'
    cfg.data.train.data_root = 'XXXX'
    cfg.data.train.ann_file = f'traini{fold}.txt'
    cfg.data.train.img_prefix = 'radiographs'
    cfg.data.train.pipeline[4].mean = [136.88, 136.88, 136.88]
    cfg.data.train.pipeline[4].std = [61.59, 61.59, 61.59]
    cfg.train_pipeline[4].mean = [136.88, 136.88, 136.88]
    cfg.train_pipeline[4].std = [61.59, 61.59, 61.59]

    # modify validation set settings
    cfg.data.val.type = 'RadiographDataset'
    cfg.data.val.data_root = 'XXXX'
    cfg.data.val.ann_file = f'vali{fold}.txt'
    cfg.data.val.img_prefix = 'radiographs'
    cfg.data.val.pipeline[1].transforms[2].mean = [136.88, 136.88, 136.88]
    cfg.data.val.pipeline[1].transforms[2].std = [61.59, 61.59, 61.59]

    # modify test set settings
    cfg.data.test.type = 'RadiographDataset'
    cfg.data.test.data_root = 'XXXX'
    cfg.data.test.ann_file = f'{proposals}i{fold}.txt'
    cfg.data.test.img_prefix = 'radiographs'
    cfg.data.test.pipeline[1].transforms[2].mean = [136.88, 136.88, 136.88]
    cfg.data.test.pipeline[1].transforms[2].std = [61.59, 61.59, 61.59]
    cfg.data.test.pipeline[1].img_scale = [
        (1333, 640), (1333, 672), (1333, 704),
        (1333, 736), (1333, 768), (1333, 800),
    ]
    cfg.data.test.pipeline[1].flip = True
    cfg.test_pipeline[1].transforms[2].mean = [136.88, 136.88, 136.88]
    cfg.test_pipeline[1].transforms[2].std = [61.59, 61.59, 61.59]
    cfg.test_pipeline[1].img_scale = [
        (1333, 640), (1333, 672), (1333, 704),
        (1333, 736), (1333, 768), (1333, 800),
    ]
    cfg.test_pipeline[1].flip = True


    ############################################################################
    # Model settings                                                           #
    ############################################################################
    # modify number of classes of the model in box head

    # load model from checkpoint to either start or resume training with
    cfg.load_from = load_from


    ############################################################################
    # Training settings                                                        #
    ############################################################################
    # train for 12 epochs, dividing LR by 10 afer epochs 8 and 11
    # cfg.checkpoint_config = None

    cfg.optimizer.lr = 0.00005
    cfg.optimizer.weight_decay = 0.1
    cfg.lr_config.warmup = None
    cfg.lr_config.step = [12, 8, 11]
    cfg.runner.max_epochs = 12

    cfg.data.samples_per_gpu = 2


    # cfg.lr_config.step = [5, 7]
    # cfg.runner.max_epochs = 8


    cfg.lr_config.step = [16, 22]
    cfg.runner.max_epochs = 24

    # only one GPU is used
    cfg.device = 'cuda'
    cfg.gpu_ids = range(1)


    ############################################################################
    # Miscellaneous                                                            #
    ############################################################################    
    # set up working dir to save checkpoint and logs.
    cfg.work_dir = work_dir + f'/{fold}_rpn'
    mmcv.mkdir_or_exist(cfg.work_dir)

    # show train results every 10 batches
    cfg.log_config.interval = 10

    # show validation mAP results and save checkpoint every 1 epoch
    # cfg.evaluation.metric = 'mAP'
    cfg.evaluation.interval = 100
    # cfg.evaluation.save_best = 'mAP'
    cfg.checkpoint_config.interval = 1
    cfg.checkpoint_config.max_keep_ckpts = 1


    cfg.mp_start_method = 'spawn'

    # set seed so the results are reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)

    # initialize logger and show final configuration
    print(f'Config:\n{cfg.pretty_text}')

    return cfg
    


def vit_config(
    checkpoint_file='checkpoints/vit-base-p32_3rdparty_pt-64xb64_in1k-224_20210928-eee25dd4.pth',
    work_dir='vit',
    resume_from=None,
    classes=2,
    idx=-1,
):
    cfg = Config.fromfile('/home/mka3dlab/Lesions/mmclassification/configs/vision_transformer/vit-base-p32_ft-64xb64_in1k-384.py')

    # Modify the number of classes in the head.
    cfg.model.head.num_classes = classes
    # cfg.model.train_cfg.augments[0].num_classes = 2
    # cfg.model.train_cfg.augments[1].num_classes = 2
    cfg.model.head.topk = (1, )

    # Load the pre-trained model's checkpoint.
    cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone')

    # Specify sample size and number of workers.
    cfg.data.samples_per_gpu = 12
    cfg.data.workers_per_gpu = 2

    # Specify the path and meta files of training dataset
    cfg.data.train.data_prefix = ''
    cfg.data.train.ann_file = 'data/classify_train.txt'
    cfg.data.train.classes = 'data/classes.txt'

    # Specify the path and meta files of validation dataset
    cfg.data.val.data_prefix = ''
    cfg.data.val.ann_file = 'data/classify_val.txt'
    cfg.data.val.classes = 'data/classes.txt'

    # Specify the path and meta files of test dataset
    cfg.data.test.data_prefix = ''
    cfg.data.test.ann_file = 'data/classify_test.txt'
    cfg.data.test.classes = 'data/classes.txt'

    # Specify the normalization parameters in data pipeline
    normalize_cfg = dict(type='Normalize', mean=[136.88, 136.88, 136.88], std=[61.59, 61.59, 61.59], to_rgb=True)
    cfg.data.train.pipeline[3] = normalize_cfg
    cfg.data.val.pipeline[3] = normalize_cfg
    cfg.data.test.pipeline[3] = normalize_cfg
    cfg['train_pipeline'][3] = normalize_cfg
    cfg['test_pipeline'][3] = normalize_cfg
    cfg['img_norm_cfg'] = normalize_cfg

    # Modify the evaluation metric
    cfg.evaluation['metric_options'] = {'topk': (1, )}
    cfg.evaluation.interval = 1

    # Specify the optimizer
    # default 0.015, 0.02 worse, 0.01 at least as good, but around worse results
    cfg.optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0001)
    cfg.optimizer_config = dict(grad_clip=None)

    # Specify the learning rate scheduler
    # cfg.lr_config = dict(policy='step', step=[9, 11], gamma=0.1)
    # cfg.runner = dict(type='EpochBasedRunner', max_epochs=12)

    # Specify the work directory
    cfg.work_dir = work_dir
    cfg.resume_from = resume_from

    # Output logs for every 10 iterations
    cfg.log_config.interval = 10

    # Set the random seed and enable the deterministic option of cuDNN
    # to keep the results' reproducible.
    cfg.seed = 0
    set_random_seed(0, deterministic=True)

    cfg.gpu_ids = range(1)

    return cfg

def swin_config(
    checkpoint_file='checkpoints/swin-base_3rdparty_in21k.pth',
    work_dir='swin',
    resume_from=None,
    classes=2,
    idx=-1,
):
    cfg = Config.fromfile('mmclassification/configs/swin_transformer/swin-base_16xb64_in1k.py')
    
    # Modify the number of classes in the head.
    cfg.model.head.num_classes = classes
    cfg.model.train_cfg.augments[0].num_classes = classes
    cfg.model.train_cfg.augments[1].num_classes = classes
    cfg.model.head.topk = (1, )

     # Load the pre-trained model's checkpoint.
    cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone')

    # Specify sample size and number of workers.
    cfg.data.samples_per_gpu = 12
    cfg.data.workers_per_gpu = 2

    # Specify the path and meta files of training dataset
    cfg.data.train.data_prefix = ''
    cfg.data.train.ann_file = 'data/classify_train.txt'
    cfg.data.train.classes = 'data/classes.txt'

    # Specify the path and meta files of validation dataset
    cfg.data.val.data_prefix = ''
    cfg.data.val.ann_file = 'data/classify_val.txt'
    cfg.data.val.classes = 'data/classes.txt'

    # Specify the path and meta files of test dataset
    cfg.data.test.data_prefix = ''
    cfg.data.test.ann_file = 'data/classify_test.txt'
    cfg.data.test.classes = 'data/classes.txt'

    # Specify the normalization parameters in data pipeline
    normalize_cfg = dict(type='Normalize', mean=[88.45, 100.93, 165.92], std=[41.24, 43.66, 39.02], to_rgb=True)
    cfg.data.train.pipeline[4]['fill_color'] =[88.45, 100.93, 165.92]
    cfg.data.train.pipeline[4]['fill_std'] = [41.24, 43.66, 39.02]
    cfg.data.train.pipeline[5] = normalize_cfg
    cfg.data.val.pipeline[3] = normalize_cfg
    cfg.data.test.pipeline[3] = normalize_cfg
    cfg['train_pipeline'][4]['fill_color'] =[88.45, 100.93, 165.92]
    cfg['train_pipeline'][4]['fill_std'] = [41.24, 43.66, 39.02]
    cfg['train_pipeline'][5] = normalize_cfg
    cfg['test_pipeline'][3] = normalize_cfg
    cfg['img_norm_cfg'] = normalize_cfg

    # Modify the evaluation metric
    cfg.evaluation['metric_options'] = {'topk': (1, )}
    cfg.evaluation.interval = 1

    # Specify the optimizer
    # default = 0.005, 0.01 worse, 0.002 better, 0.001 better
    cfg.optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
    cfg.optimizer_config = dict(grad_clip=None)

    # Specify the work directory
    cfg.work_dir = work_dir
    cfg.resume_from = resume_from

    # Output logs for every 10 iterations
    cfg.log_config.interval = 10

    # Set the random seed and enable the deterministic option of cuDNN
    # to keep the results' reproducible.
    cfg.seed = 0
    set_random_seed(0, deterministic=True)

    cfg.gpu_ids = range(1)

    return cfg



def vitadapter_det_config(
    load_from=None,
    resume_from=None,
    work_dir='result',
    fold: int=-1,
    version='',
    affix=''
):
    # load Faster R-CNN configuration with swin model
    cfg_prefix = 'mmdetection/configs/'
    cfg = Config.fromfile(
        cfg_prefix + 'faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py',
    )
    
    fold = '' if fold == -1 else str(fold)

    ############################################################################
    # Data settings                                                            #
    ############################################################################
    # modify general dataset settings
    cfg.dataset_type = 'RadiographDataset'
    cfg.data_root = 'XXXX'
    cfg.img_norm_cfg.mean = [136.88, 136.88, 136.88]
    cfg.img_norm_cfg.std = [61.59, 61.59, 61.59]

    # modify train set settings
    cfg.data.train.type = 'RadiographDataset'
    cfg.data.train.data_root = 'XXXX'
    cfg.data.train.ann_file = f'traini{fold}{affix}.txt'
    cfg.data.train.img_prefix = 'radiographs'
    cfg.data.train.pipeline[4].mean = [136.88, 136.88, 136.88]
    cfg.data.train.pipeline[4].std = [61.59, 61.59, 61.59]
    cfg.train_pipeline[4].mean = [136.88, 136.88, 136.88]
    cfg.train_pipeline[4].std = [61.59, 61.59, 61.59]
    cfg.data.train.pipeline[2].img_scale = [
        # (1333, 640), (1333, 672), (1333, 704),
        # (1333, 736), (1333, 768), (1333, 800),
        (1400,700)
    ]
    cfg.train_pipeline[2].img_scale = [
        # (1333, 640), (1333, 672), (1333, 704),
        # (1333, 736), (1333, 768), (1333, 800),
        (1400,700)
    ]

    # modify validation set settings
    cfg.data.val.type = 'RadiographDataset'
    cfg.data.val.data_root = 'XXXX'
    cfg.data.val.ann_file = f'vali{fold}{affix}.txt'
    cfg.data.val.img_prefix = 'radiographs'
    cfg.data.val.pipeline[1].transforms[2].mean = [136.88, 136.88, 136.88]
    cfg.data.val.pipeline[1].transforms[2].std = [61.59, 61.59, 61.59]
    cfg.data.val.pipeline[1].img_scale = [
        # (1333, 640), (1333, 672), (1333, 704),
        # (1333, 736), (1333, 768), (1333, 800),
        (1400,700)
    ]

    # modify test set settings
    cfg.data.test.type = 'RadiographDataset'
    cfg.data.test.data_root = 'XXXX'
    cfg.data.test.ann_file = f'testi{fold}{affix}.txt'
    cfg.data.test.img_prefix = 'radiographs'
    cfg.data.test.pipeline[1].transforms[2].mean = [136.88, 136.88, 136.88]
    cfg.data.test.pipeline[1].transforms[2].std = [61.59, 61.59, 61.59]
    cfg.data.test.pipeline[1].img_scale = [
        # (1333, 640), (1333, 672), (1333, 704),
        # (1333, 736), (1333, 768), (1333, 800),
        (1400,700)
    ]
    cfg.data.test.pipeline[1].flip = True
    cfg.test_pipeline[1].transforms[2].mean = [136.88, 136.88, 136.88]
    cfg.test_pipeline[1].transforms[2].std = [61.59, 61.59, 61.59]
    cfg.test_pipeline[1].img_scale = [
        # (1333, 640), (1333, 672), (1333, 704),
        # (1333, 736), (1333, 768), (1333, 800),
        (1400,700)
    ]
    cfg.test_pipeline[1].flip = True

    ############################################################################
    # Model settings                                                           #
    ############################################################################
    # modify number of classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = len(RadiographDataset.CLASSES)

    # load model from checkpoint to either start or resume training with
    cfg.load_from = load_from
    cfg.resume_from = resume_from


    ############################################################################
    # Training settings                                                        #
    ############################################################################
    # train for 12 epochs, dividing LR by 10 afer epochs 8 and 11
    cfg.checkpoint_config = None

    cfg.optimizer.lr = 0.00005
    cfg.optimizer.weight_decay = 0.1
    cfg.lr_config.warmup = None
    cfg.lr_config.step = [12, 8, 11]
    cfg.runner.max_epochs = 12

    cfg.data.samples_per_gpu = 2

    # cfg.lr_config.step = [5, 7]
    # cfg.runner.max_epochs = 8

    cfg.lr_config.step = [16, 22]
    cfg.runner.max_epochs = 24

    # only one GPU is used
    cfg.device = 'cuda'
    cfg.gpu_ids = range(1)

    ############################################################################
    # Miscellaneous                                                            #
    ############################################################################    
    # set up working dir to save checkpoint and logs.
    # cfg.work_dir = work_dir + f'/{fold}_resnet'
    cfg.work_dir = work_dir + f'/fold_{fold}{version}'
    mmcv.mkdir_or_exist(cfg.work_dir)

    # show train results every 10 batches
    cfg.log_config.interval = 10

    # show validation mAP results and save checkpoint every 1 epoch
    cfg.evaluation.metric = 'mAP'
    cfg.evaluation.interval = 1
    cfg.evaluation.save_best = 'mAP'
    # cfg.checkpoint_config.interval = 1
    
    # 新增配置
    cfg.checkpoint_config = dict(
                                interval=1,
                                max_keep_ckpts=-1,
                                create_symlink=True
                                )
    

    cfg.mp_start_method = 'spawn'

    # set seed so the results are reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)

    # initialize logger and show final configuration
    print(f'Config:\n{cfg.pretty_text}')

    return cfg


def vitsplit_det_config(
    load_from=None,
    resume_from=None,
    work_dir='result',
    fold: int=-1,
    version='',
    affix=''         
):
    # load Faster R-CNN configuration with swin model
    cfg_prefix = 'mmdetection/configs/'
    cfg = Config.fromfile(
        cfg_prefix + 'faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py',
    )
    
    fold = '' if fold == -1 else str(fold)

    ############################################################################
    # Data settings                                                            #
    ############################################################################
    # modify general dataset settings
    cfg.dataset_type = 'RadiographDataset'
    cfg.data_root = 'XXXX'
    cfg.img_norm_cfg.mean = [136.88, 136.88, 136.88]
    cfg.img_norm_cfg.std = [61.59, 61.59, 61.59]

    # modify train set settings
    cfg.data.train.type = 'RadiographDataset'
    cfg.data.train.data_root = 'XXXX'
    cfg.data.train.ann_file = f'traini{fold}{affix}.txt'
    cfg.data.train.img_prefix = 'radiographs'
    cfg.data.train.pipeline[4].mean = [136.88, 136.88, 136.88]
    cfg.data.train.pipeline[4].std = [61.59, 61.59, 61.59]
    cfg.train_pipeline[4].mean = [136.88, 136.88, 136.88]
    cfg.train_pipeline[4].std = [61.59, 61.59, 61.59]
    cfg.data.train.pipeline[2].img_scale = [
        # (1333, 640), (1333, 672), (1333, 704),
        # (1333, 736), (1333, 768), (1333, 800),
        (1400,700)
    ]
    cfg.train_pipeline[2].img_scale = [
        # (1333, 640), (1333, 672), (1333, 704),
        # (1333, 736), (1333, 768), (1333, 800),
        (1400,700)
    ]

    # modify validation set settings
    cfg.data.val.type = 'RadiographDataset'
    cfg.data.val.data_root = 'XXXX'
    cfg.data.val.ann_file = f'vali{fold}{affix}.txt'
    cfg.data.val.img_prefix = 'radiographs'
    cfg.data.val.pipeline[1].transforms[2].mean = [136.88, 136.88, 136.88]
    cfg.data.val.pipeline[1].transforms[2].std = [61.59, 61.59, 61.59]
    cfg.data.val.pipeline[1].img_scale = [
        # (1333, 640), (1333, 672), (1333, 704),
        # (1333, 736), (1333, 768), (1333, 800),
        (1400,700)
    ]

    # modify test set settings
    cfg.data.test.type = 'RadiographDataset'
    cfg.data.test.data_root = 'XXXX'
    cfg.data.test.ann_file = f'testi{fold}{affix}.txt'
    cfg.data.test.img_prefix = 'radiographs'
    cfg.data.test.pipeline[1].transforms[2].mean = [136.88, 136.88, 136.88]
    cfg.data.test.pipeline[1].transforms[2].std = [61.59, 61.59, 61.59]
    cfg.data.test.pipeline[1].img_scale = [
        # (1333, 640), (1333, 672), (1333, 704),
        # (1333, 736), (1333, 768), (1333, 800),
        (1400,700)
    ]
    cfg.data.test.pipeline[1].flip = True
    cfg.test_pipeline[1].transforms[2].mean = [136.88, 136.88, 136.88]
    cfg.test_pipeline[1].transforms[2].std = [61.59, 61.59, 61.59]
    cfg.test_pipeline[1].img_scale = [
        # (1333, 640), (1333, 672), (1333, 704),
        # (1333, 736), (1333, 768), (1333, 800),
        (1400,700)
    ]
    cfg.test_pipeline[1].flip = True

    ############################################################################
    # Model settings                                                           #
    ############################################################################
    # modify number of classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = len(RadiographDataset.CLASSES)

    # load model from checkpoint to either start or resume training with
    cfg.load_from = load_from
    cfg.resume_from = resume_from


    ############################################################################
    # Training settings                                                        #
    ############################################################################
    # train for 12 epochs, dividing LR by 10 afer epochs 8 and 11
    # cfg.checkpoint_config = None
    # 新增配置
    cfg.checkpoint_config = dict(
                                interval=1,
                                max_keep_ckpts=-1,
                                create_symlink=True
                                )

    cfg.optimizer.lr = 0.00005
    cfg.optimizer.weight_decay = 0.1
    cfg.lr_config.warmup = None
    cfg.lr_config.step = [12, 8, 11]
    cfg.runner.max_epochs = 12

    cfg.data.samples_per_gpu = 2

    # cfg.lr_config.step = [5, 7]
    # cfg.runner.max_epochs = 8

    cfg.lr_config.step = [16, 22]
    cfg.runner.max_epochs = 24

    # only one GPU is used
    cfg.device = 'cuda'
    cfg.gpu_ids = range(1)

    ############################################################################
    # Miscellaneous                                                            #
    ############################################################################    
    # set up working dir to save checkpoint and logs.
    # cfg.work_dir = work_dir + f'/{fold}_resnet'
    cfg.work_dir = work_dir + f'/fold_{fold}{version}'
    mmcv.mkdir_or_exist(cfg.work_dir)

    # show train results every 10 batches
    cfg.log_config.interval = 10

    # show validation mAP results and save checkpoint every 1 epoch
    cfg.evaluation.metric = 'mAP'
    cfg.evaluation.interval = 1
    cfg.evaluation.save_best = 'mAP'
    # cfg.checkpoint_config.interval = 1

    cfg.mp_start_method = 'spawn'

    # set seed so the results are reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)

    # initialize logger and show final configuration
    print(f'Config:\n{cfg.pretty_text}')

    return cfg