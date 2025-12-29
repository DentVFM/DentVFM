# Author: Tony Xu
#
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

from dinov2.data.loaders import make_segmentation_dataset_3d
from dinov2.data import SamplerType, make_data_loader
from dinov2.eval.segmentation_3d.segmentation_heads import UNETRHead, LinearDecoderHead, ViTAdapterUNETRHead, UNETRHeaforSwim
from dinov2.eval.setup import get_args_parser, setup_and_build_model_3d, setup_and_build_model_sammed3d, setup_and_build_model_m3d, setup_and_build_model_swimunetr
from dinov2.eval.segmentation_3d.augmentations import make_transforms, make_transforms_sammed3d, make_transforms_m3d, make_transforms_swimunetr
from dinov2.eval.segmentation_3d.metrics import get_metric
import os
import torch
import json
import sys
import shutil
from functools import partial
from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
from monai.data.utils import list_data_collate
from monai.optimizers import WarmupCosineSchedule


def add_seg_args(parser):
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name of finetuning dataset",
    )
    parser.add_argument(
        "--dataset-percent",
        type=int,
        help="Percent of finetuning dataset to use",
        default=100
    )
    parser.add_argument(
        "--base-data-dir",
        type=str,
        help="Base data directory for finetuning dataset",
    )
    parser.add_argument(
        "--segmentation-head",
        type=str,
        help="Segmentation head",
    )
    parser.add_argument(
        "--train-feature-model",
        action="store_true",
        help="Freeze feature model or not",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Total epochs",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Iterations to perform per epoch",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        help="Iterations to perform per evaluation",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="Image side length",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        help="Scale factor for resizing images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="path to cache directory for monai persistent dataset"
    )
    # 指定modelname
    parser.add_argument(
        "--model_name",
        type=str,
        help="backbone model name, can be 'sammed3d' ",
    )
    # 是否删除已经存在的eval结果
    parser.add_argument(
        "--save_exist",
        action="store_true",
        help="do not remove existing dir",
    )

    return parser


def train_iter(model, batch, optimizer, scheduler, loss_function, scaler):
    x, y = (batch["image"].cuda(), batch["label"].cuda())
    logits = model(x)
    loss = loss_function(logits, y)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.item()


def val_iter(model, batch, metric, image_size, batch_size, overlap=0.5):
    x, y = (batch["image"].cuda(), batch["label"].cuda())
    # debug
    # print(x.shape)
    logits = sliding_window_inference(x, image_size, batch_size, model, overlap=overlap)

    iter_metric = metric(logits, y)
    return iter_metric


def do_finetune(feature_model, autocast_dtype, args, train_transforms, val_transforms):

    # get transforms, dataset, dataloaders
    # train_transforms, val_transforms = make_transforms(
    #     args.dataset_name,
    #     args.image_size,
    #     args.resize_scale,
    #     min_int=-1.0
    # )

    train_ds, val_ds, test_ds, input_channels, num_classes = make_segmentation_dataset_3d(
        args.dataset_name,
        args.dataset_percent,
        args.base_data_dir,
        train_transforms,
        val_transforms,
        args.cache_dir,
        args.batch_size
    )
    train_loader = make_data_loader(
        dataset=train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=0,
        sampler_type=SamplerType.SHARDED_INFINITE,
        drop_last=False,
        persistent_workers=True,
        collate_fn=list_data_collate
    )
    val_loader = make_data_loader(
        dataset=val_ds,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        seed=0,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
        collate_fn=list_data_collate
    )
    test_loader = make_data_loader(
        dataset=test_ds,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        seed=0,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
        collate_fn=list_data_collate
    )

    # get model
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    scaler = torch.cuda.amp.GradScaler()
    if args.segmentation_head == 'UNETR':
        # 此处修改
        if args.model_name == 'swimunetr':
            seg_model = UNETRHeaforSwim(feature_model, input_channels, args.image_size, num_classes, autocast_ctx)
        else:
            # 此处修改，添加一些模型特定参数
            if args.model_name == 'sammed3d':
                layer_idx=[2, 5, 8, 11]
                patch_size=[16,16,16]
            elif args.model_name == 'm3d':
                layer_idx=[2, 5, 8, 11]
                patch_size=[16,16,16]
            else:
                layer_idx=[5, 11, 17, 23]
                patch_size=[16,16,16]
            seg_model = UNETRHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx, layer_idx=layer_idx, patch_size=patch_size)
    
    elif args.segmentation_head == 'Linear':
        seg_model = LinearDecoderHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx)
    elif args.segmentation_head == 'ViTAdapterUNETR':
        seg_model = ViTAdapterUNETRHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx)
    else:
        raise ValueError(f"Unknown segmentation head: {args.segmentation_head}")

    if args.train_feature_model:
        if args.segmentation_head == 'ViTAdapterUNETR':
            seg_model.feature_model.vit_model.train()
        else:
            seg_model.feature_model.train()

    else:
        if args.segmentation_head == 'ViTAdapterUNETR':
            seg_model.feature_model.vit_model.eval()
            for param in seg_model.feature_model.vit_model.parameters():
                param.requires_grad = False
        else:
            seg_model.feature_model.eval()
            for param in seg_model.feature_model.parameters():
                param.requires_grad = False

    trainable_params = [name for name, param in seg_model.named_parameters() if param.requires_grad]
    print(f"Trainable parameters: {trainable_params}")

    # get optimizer, scheduler, loss function, metric
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, seg_model.parameters()), lr=args.learning_rate)
    max_iter = args.epochs * args.epoch_length
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=args.warmup_iters,
        t_total=max_iter
    )

    if args.dataset_name == 'BTCV' or args.dataset_name == 'LA-SEG' or args.dataset_name == 'TDSC-ABUS' or ('Dataset114_Sts_miccia23' in args.dataset_name) or ('Dataset115_Pal' in args.dataset_name) or ('Dataset116_HeadNeck_miccai15' in args.dataset_name) or ('Dataset117_Pulpy3D' in args.dataset_name):
        # 此处修改
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        # ce_weight = torch.tensor([1.0 for _ in range(49)])
        # ce_weight[0] = 0.1
        # loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=ce_weight)
    elif 'Dataset112_ToothFairy2' in args.dataset_name:
        # 此处修改
        # loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        ce_weight = torch.tensor([1.0 for _ in range(49)])
        ce_weight[0] = 0.1
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=ce_weight)
    elif 'Dataset118_ToothFairy3' in args.dataset_name:
        # 此处修改
        ce_weight = torch.tensor([1.0 for _ in range(75)])
        ce_weight[0] = 0.1
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=ce_weight)
    elif 'Dataset113_NCseg' in args.dataset_name:
        # 此处修改
        ce_weight = torch.tensor([0.1, 1.0])
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=ce_weight)
    elif args.dataset_name == 'BraTS':
        loss_fn = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")

    dice_metric = get_metric(args.dataset_name)

    seg_model.cuda()
    loss_fn.cuda()

    best_val_dice = -1
    train_loss_sum = 0
    iters_list = []
    train_loss_list = []
    val_dice_list = []
    val_per_cls_dice_list = []

    for it, train_data in enumerate(train_loader):

        # train for one iteration
        train_loss = train_iter(
            model=seg_model,
            batch=train_data,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_fn,
            scaler=scaler
        )
        train_loss_sum += train_loss

        if it % 100 == 0:
            print(f"[Iter {it}], Train loss: {train_loss}", flush=True)

        if it % args.eval_iters == 0:
            # valdation
            total_val_dice = 0
            total_per_cls_val_dice = [0 for _ in range(num_classes)]
            val_steps = 0
            seg_model.eval()
            
            
#             with torch.no_grad():
#                 for val_data in val_loader:
#                     val_dice, val_per_cls_dice = val_iter(
#                         model=seg_model,
#                         batch=val_data,
#                         image_size=(args.image_size,) * 3,
#                         batch_size=args.batch_size,
#                         metric=dice_metric,
#                         overlap=0.
#                     )

#                     total_val_dice += val_dice
#                     for i in range(num_classes):
#                         total_per_cls_val_dice[i] += val_per_cls_dice[i]
#                     val_steps += 1

#             avg_val_dice = total_val_dice / val_steps
#             avg_per_cls_val_dice = [total_per_cls_val_dice[i] / val_steps for i in range(num_classes)]
#             avg_train_loss = train_loss_sum / args.eval_iters
            # 此处修改
            total_val_steps = 0
            per_cls_steps = [0 for _ in range(num_classes)]
            with torch.no_grad():
                for val_data in val_loader:
                    val_dice, val_per_cls_dice = val_iter(
                        model=seg_model,
                        batch=val_data,
                        image_size=(args.image_size,) * 3,
                        batch_size=args.batch_size,
                        metric=dice_metric,
                        overlap=0.
                    )

                    total_val_dice += val_dice
                    total_val_steps += 1
                    for i in range(num_classes):
                        if val_per_cls_dice[i]<0:
                            total_per_cls_val_dice[i] += 0.0
                        else:
                            total_per_cls_val_dice[i] += val_per_cls_dice[i]
                            per_cls_steps[i] += 1
            # 防止除零异常
            per_cls_steps = [1.0 if x == 0.0 else x for x in per_cls_steps]
            avg_val_dice = total_val_dice / total_val_steps
            avg_per_cls_val_dice = [total_per_cls_val_dice[i] / per_cls_steps[i] for i in range(num_classes)]
            
            avg_train_loss = train_loss_sum / args.eval_iters
            

            train_loss_list.append(avg_train_loss)
            val_dice_list.append(avg_val_dice)
            val_per_cls_dice_list.append(avg_per_cls_val_dice)
            iters_list.append(it)
            train_loss_sum = 0

            print(f"[Iter {it}], Train loss: {avg_train_loss}, Val dice: {avg_val_dice}")
            print(f"Val per class dice: {avg_per_cls_val_dice}")

            # save best model
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                print(f"Saving best model with val dice: {best_val_dice} on iter: {it}")
                torch.save(seg_model.state_dict(), args.output_dir + "/best_model.pth")

            # set back to train mode
            seg_model.train()
            if not args.train_feature_model:
                if args.segmentation_head == 'ViTAdapterUNETR':
                    seg_model.feature_model.vit_model.eval()
                else:
                    seg_model.feature_model.eval()

        if it >= max_iter:
            break

    # test
    seg_model.load_state_dict(torch.load(args.output_dir + "/best_model.pth"))
    seg_model.eval()

    total_test_dice = 0
    total_per_cls_test_dice = [0 for _ in range(num_classes)]
    test_steps = 0
    seg_model.eval()
    
#     with torch.no_grad():
#         for test_data in test_loader:
#             test_dice, test_per_cls_dice = val_iter(
#                 model=seg_model,
#                 batch=test_data,
#                 image_size=(args.image_size,) * 3,
#                 batch_size=args.batch_size,
#                 metric=dice_metric,
#                 overlap=0.75
#             )

#             total_test_dice += test_dice
#             for i in range(num_classes):
#                 total_per_cls_test_dice[i] += test_per_cls_dice[i]
#             test_steps += 1
#     avg_test_dice = total_test_dice / test_steps
#     avg_per_cls_test_dice = [total_per_cls_test_dice[i] / test_steps for i in range(num_classes)]
    
    # 此处修改计算方式，将nan情况进行过滤
    total_test_step = 0
    per_cls_test_step = [0 for i in range(num_classes)]
    with torch.no_grad():
        for test_data in test_loader:
            test_dice, test_per_cls_dice = val_iter(
                model=seg_model,
                batch=test_data,
                image_size=(args.image_size,) * 3,
                batch_size=args.batch_size,
                metric=dice_metric,
                overlap=0.75
            )
            total_test_dice += test_dice
            total_test_step += 1
            for i in range(num_classes):
                if test_per_cls_dice[i]<0:
                    total_per_cls_test_dice[i] += 0.0
                else:
                    total_per_cls_test_dice[i] += test_per_cls_dice[i]
                    per_cls_test_step[i] += 1
    # 防止除零异常
    per_cls_test_step = [1.0 if x == 0.0 else x for x in per_cls_test_step]
    avg_test_dice = total_test_dice / total_test_step
    avg_per_cls_test_dice = [total_per_cls_test_dice[i] / per_cls_test_step[i] for i in range(num_classes)]
    
    
    # 记录val指标
    print(f"Test dice: {avg_test_dice}")
    print(f"Test per class dice: {avg_per_cls_test_dice}")

    with open(f'{args.output_dir}/results.json', 'w') as fp:
        json.dump({
            'iters_list': iters_list,
            'train_loss_list': train_loss_list,
            'val_dice_list': val_dice_list,
            'val_per_cls_dice_list': val_per_cls_dice_list,
            'test_dice': avg_test_dice,
            'test_per_cls_dice': avg_per_cls_test_dice,
        }, fp)


def main(args):
    # feature_model, autocast_dtype = setup_and_build_model_3d(args)
    if args.model_name == 'sammed3d':
        print("当前评测的模型为：", args.model_name)
        # use sammed3d model
        feature_model, autocast_dtype = setup_and_build_model_sammed3d(args)
        train_transforms, val_transforms = make_transforms_sammed3d(
            args.dataset_name,
            args.image_size,
            args.resize_scale,
            min_int=0
        )
    elif args.model_name == 'm3d':
        print("当前评测的模型为：", args.model_name)
        # use sammed3d model
        feature_model, autocast_dtype = setup_and_build_model_m3d(args)
        train_transforms, val_transforms = make_transforms_m3d(
            args.dataset_name,
            args.image_size,
            args.resize_scale,
            min_int=0,
        )
    elif args.model_name == 'swimunetr':
        print("当前评测的模型为：", args.model_name)
        feature_model, autocast_dtype = setup_and_build_model_swimunetr(args)
        train_transforms, val_transforms = make_transforms_swimunetr(
            args.dataset_name,
            args.image_size,
            args.resize_scale,
            min_int=0,
        )
    else:
        print("当前评测的模型为：", args.model_name)
        feature_model, autocast_dtype = setup_and_build_model_3d(args)
        train_transforms, val_transforms = make_transforms(
            args.dataset_name,
            args.image_size,
            args.resize_scale,
            min_int=-1.0
        )

    do_finetune(feature_model, autocast_dtype, args, train_transforms, val_transforms)


if __name__ == "__main__":
    args = add_seg_args(get_args_parser(add_help=True)).parse_args()

    # 判断输出路径是否存在，如果存在则先删除目录
    if os.path.exists(args.output_dir) and os.path.isdir(args.output_dir):
        print("Find existing dir!")
        if os.path.exists(os.path.join(args.output_dir,'results.json')) and args.save_exist:
            print("Save existing do not reevaluation!")
            sys.exit(0)
        else:
            print("Start removing this dir!")
            for item in os.listdir(args.output_dir):
                item_path = os.path.join(args.output_dir, item)
                if os.path.isdir(item_path):
                    # 如果是子目录，则递归删除
                    shutil.rmtree(item_path)
                else:
                    # 如果是文件，直接删除
                    os.remove(item_path)

    main(args)