# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import gc
import logging
import sys
import time
from typing import List, Optional
import os
import json
import shutil
from cuml.linear_model import LogisticRegression
import torch
import torch.backends.cudnn as cudnn
import torch.distributed
from torch import nn
from torch.utils.data import TensorDataset
from torchmetrics import MetricTracker

from dinov2.data import make_dataset
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.distributed import get_global_rank, get_global_size
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model,setup_and_build_model_resnet50,setup_and_build_model_biomedclip,setup_and_build_model_clip,setup_and_build_model_sam,setup_and_build_model_lvm_resnet50,setup_and_build_model_lvm_vitb16, setup_and_build_model_sammed2d, setup_and_build_model_dinov3
from dinov2.eval.utils import evaluate, extract_features
from dinov2.utils.dtype import as_torch_dtype
from torchvision import transforms
import numpy as np


logger = logging.getLogger("dinov2")

DEFAULT_MAX_ITER = 1_0000
C_POWER_RANGE = torch.linspace(-6, 5, 45)
_CPU_DEVICE = torch.device("cpu")


# 设置logistics regression参数
def get_args_parser(
        description: Optional[str] = None,
        parents: Optional[List[argparse.ArgumentParser]] = None,
        add_help: bool = True,
    ):
    parents = parents or []
    # 基础设置：配置文件、模型权重、eval结果输出路径、其他等
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
    )
    # 此处添加控制提特征时使用的batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--finetune-dataset-str",
        dest="finetune_dataset_str",
        type=str,
        help="Fine-tuning dataset",
    )
    parser.add_argument(
        "--finetune-on-val",
        action="store_true",
        help="If there is no finetune dataset, whether to choose the "
        "hyperparameters on the val set instead of 10%% of the train dataset",
    )
    parser.add_argument(
        "--metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Metric type",
    )
    parser.add_argument(
        "--train-features-device",
        type=str,
        help="Device to gather train features (cpu, cuda, cuda:0, etc.), default: %(default)s",
    )
    parser.add_argument(
        "--train-dtype",
        type=str,
        help="Data type to convert the train features to (default: %(default)s)",
    )
    parser.add_argument(
        "--max-train-iters",
        type=int,
        help="Maximum number of train iterations (default: %(default)s)",
    )
    # 此处添加命令行输入的数据集划分大小
    parser.add_argument(
        "--train_val_test_nums",
        nargs="+",
        type=int,
        help="dataset split number",
    )
    # 指定modelname
    parser.add_argument(
        "--model_name",
        type=str,
        help="backbone model name, can be 'resnet50', 'dinov2', 'biomedclip'",
    )
    parser.add_argument(
        "--model_version_name",
        type=str,
        help="specific model version name",
    )
    # 是否删除已经存在的eval结果
    parser.add_argument(
        "--save_exist",
        action="store_true",
        help="do not remove existing dir",
    )
    # 图像尺寸大小，用于控制our的resize和crop，可以方便扩增到512分辨率
    parser.add_argument(
        "--image_res",
        type=int,
        help="image resolution, default: 224",
    )
    parser.set_defaults(
        model_name='dinov2',
        model_version_name='training_vitb16',
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        finetune_dataset_str=None,
        metric_type=MetricType.MEAN_ACCURACY,
        train_features_device="cpu",
        train_dtype="float64",
        max_train_iters=DEFAULT_MAX_ITER,
        finetune_on_val=False,
        batch_size=256,
        train_val_test_nums=[100, 100, 100],
        image_res=224,
    )
    return parser


# 数据预处理方法
class BiomedClipTransform:
    def __init__(self, preprocess):
        self.preprocess = preprocess
    def __call__(self, x):
        return self.preprocess(x)

class ClipTransform:
    def __init__(self, preprocess):
        self.preprocess = preprocess
    def __call__(self, x):
        x_t = self.preprocess(images=x, return_tensors="pt")['pixel_values']    # (1,3,w,h)
        
        return torch.squeeze(x_t)   # (3,w,h)

class SAMTransform:
    def __init__(self, preprocess):
        self.preprocess = preprocess
    def __call__(self, x):
        input_points = [[[1, 1]]]
        out_image = self.preprocess(x, input_points=input_points, return_tensors="pt")['pixel_values']
        return torch.squeeze(out_image)
    

# 封装logreg
class LogRegModule(nn.Module):
    def __init__(
        self,
        C,
        max_iter=DEFAULT_MAX_ITER,
        dtype=torch.float64,
        device=_CPU_DEVICE,
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.estimator = LogisticRegression(
            penalty="l2",
            C=C,
            max_iter=max_iter,
            output_type="numpy",
            tol=1e-12,
            linesearch_max_iter=50,
            class_weight='balanced',
        )

    def forward(self, samples, targets):
        samples_device = samples.device
        samples = samples.to(dtype=self.dtype, device=self.device)
        if self.device == _CPU_DEVICE:
            samples = samples.numpy()
        probas = self.estimator.predict_proba(samples)
        return {"preds": torch.from_numpy(probas).to(samples_device), "target": targets}

    def fit(self, train_features, train_labels):
        train_features = train_features.to(dtype=self.dtype, device=self.device)
        # train_labels = train_labels.to(dtype=self.dtype, device=self.device)
        train_labels = train_labels.to(dtype=torch.int32, device=self.device)
        if self.device == _CPU_DEVICE:
            # both cuML and sklearn only work with numpy arrays on CPU
            train_features = train_features.numpy()
            train_labels = train_labels.numpy()
        self.estimator.fit(train_features, train_labels)


# 执行评价
def evaluate_model(*, logreg_model, logreg_metric, test_data_loader, device):
    postprocessors = {"metrics": logreg_model}
    metrics = {"metrics": logreg_metric}
    return evaluate(nn.Identity(), test_data_loader, postprocessors, metrics, device)


# 给超参数，训一个logreg
def train_for_C(*, C, max_iter, train_features, train_labels, dtype=torch.float64, device=_CPU_DEVICE):
    logreg_model = LogRegModule(C, max_iter=max_iter, dtype=dtype, device=device)
    logreg_model.fit(train_features, train_labels)
    return logreg_model


# 训练并评价模型
def train_and_evaluate(
        *,
        C,
        max_iter,
        train_features,
        train_labels,
        logreg_metric,
        test_data_loader,
        train_dtype=torch.float64,
        train_features_device,
        eval_device,
    ):
    # 训练模型
    logreg_model = train_for_C(
        C=C,
        max_iter=max_iter,
        train_features=train_features,
        train_labels=train_labels,
        dtype=train_dtype,
        device=train_features_device,
    )
    # 返回评价结果
    return evaluate_model(
        logreg_model=logreg_model,
        logreg_metric=logreg_metric,
        test_data_loader=test_data_loader,
        device=eval_device,
    )


# 对不同超参数进行实验，取最好的超参数
def sweep_C_values(
        *,
        train_features,
        train_labels,
        test_data_loader,
        metric_type,
        num_classes,
        train_dtype=torch.float64,
        train_features_device=_CPU_DEVICE,
        max_train_iters=DEFAULT_MAX_ITER,
    ):
    if metric_type == MetricType.PER_CLASS_ACCURACY:
        # If we want to output per-class accuracy, we select the hyperparameters with mean per class
        metric_type = MetricType.MEAN_PER_CLASS_ACCURACY
    logreg_metric = build_metric(metric_type, num_classes=num_classes)
    metric_tracker = MetricTracker(logreg_metric, maximize=True)
    ALL_C = 10**C_POWER_RANGE
    logreg_models = {}

    train_features = train_features.to(dtype=train_dtype, device=train_features_device)
    train_labels = train_labels.to(device=train_features_device)
    # 将要调的参数分配到不同进程中执行，每个进程分配均衡
    for i in range(get_global_rank(), len(ALL_C), get_global_size()):
        C = ALL_C[i].item()
        logger.info(
            f"Training for C = {C:.5f}, dtype={train_dtype}, "
            f"features: {train_features.shape}, {train_features.dtype}, "
            f"labels: {train_labels.shape}, {train_labels.dtype}"
        )
        logreg_models[C] = train_for_C(
            C=C,
            max_iter=max_train_iters,
            train_features=train_features,
            train_labels=train_labels,
            dtype=train_dtype,
            device=train_features_device,
        )

    gather_list = [None for _ in range(get_global_size())]
    torch.distributed.all_gather_object(gather_list, logreg_models)

    logreg_models_gathered = {}
    for logreg_dict in gather_list:
        logreg_models_gathered.update(logreg_dict)

    for i in range(len(ALL_C)):
        metric_tracker.increment()
        C = ALL_C[i].item()
        evals = evaluate_model(
            logreg_model=logreg_models_gathered[C],
            logreg_metric=metric_tracker,
            test_data_loader=test_data_loader,
            device=torch.cuda.current_device(),
        )
        logger.info(f"Trained for C = {C:.5f}, accuracies = {evals}")

    #     best_stats, which_epoch = metric_tracker.best_metric(return_step=True)
    #     best_stats_100 = {k: 100.0 * v for k, v in best_stats.items()}
    #     if which_epoch["top-1"] == i:
    #         best_C = C
    # logger.info(f"Sweep best {best_stats_100}, best C = {best_C:.6f}")
    # return best_stats, best_C
    
    # 计算所有classifier的度量结果
    all_res = [metric.compute() for i, metric in enumerate(metric_tracker) if i != 0]
    # 根据acc，计算出一个最好的
    max_index = max(enumerate(all_res), key=lambda x: x[1]["top-1_acc_micro"])[0]
    best_stats = all_res[max_index]
    best_C = ALL_C[max_index].item()
    best_stats_100 = best_stats["top-1_acc_micro"]
    logger.info(f"Sweep best {best_stats_100}, best C = {best_C:.6f}")

    return best_stats, best_C, all_res
    


def eval_log_regression(
        *,
        model,
        train_dataset,
        val_dataset,
        finetune_dataset,
        metric_type,
        batch_size,
        num_workers,
        finetune_on_val=False,
        train_dtype=torch.float64,
        train_features_device=_CPU_DEVICE,
        max_train_iters=DEFAULT_MAX_ITER,
    ):
    """
    Implements the "standard" process for log regression evaluation:
    The value of C is chosen by training on train_dataset and evaluating on
    finetune_dataset. Then, the final model is trained on a concatenation of
    train_dataset and finetune_dataset, and is evaluated on val_dataset.
    If there is no finetune_dataset, the value of C is the one that yields
    the best results on a random 10% subset of the train dataset

    此处finetune相当于validation, val相当于test
    如果没有给出finetune, 那么默认取10%train作为finetune
    """

    start = time.time()
    
    # 提取训练数据样本的特征
    train_features, train_labels = extract_features(
        model, train_dataset, batch_size, num_workers, gather_on_cpu=(train_features_device == _CPU_DEVICE)
    )
    val_features, val_labels = extract_features(
        model, val_dataset, batch_size, num_workers, gather_on_cpu=(train_features_device == _CPU_DEVICE)
    )
    val_data_loader = torch.utils.data.DataLoader(
        TensorDataset(val_features, val_labels),
        batch_size=batch_size,
        drop_last=False,
        num_workers=0,
        persistent_workers=False,
    )

    # 确定finetune dataset，并提取特征
    if finetune_dataset is None and finetune_on_val:
        logger.info("Choosing hyperparameters on the val dataset")
        finetune_features, finetune_labels = val_features, val_labels   # 此时直接用val进行调参
    elif finetune_dataset is None and not finetune_on_val:
        logger.info("Choosing hyperparameters on 10% of the train dataset")
        torch.manual_seed(0)
        indices = torch.randperm(len(train_features), device=train_features.device)
        finetune_index = indices[: len(train_features) // 10]   # 随机取前10%作为finetune dataset
        train_index = indices[len(train_features) // 10 :]  # 剩下用于训练
        finetune_features, finetune_labels = train_features[finetune_index], train_labels[finetune_index]
        train_features, train_labels = train_features[train_index], train_labels[train_index]
    else:
        logger.info("Choosing hyperparameters on the finetune dataset")
        finetune_features, finetune_labels = extract_features(
            model, finetune_dataset, batch_size, num_workers, gather_on_cpu=(train_features_device == _CPU_DEVICE)
        )

    # release the model - free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    finetune_data_loader = torch.utils.data.DataLoader(
        TensorDataset(finetune_features, finetune_labels),
        batch_size=batch_size,
        drop_last=False,
    )

    # 获得类别的数量
    if len(train_labels.shape) > 1:
        num_classes = train_labels.shape[1]
    else:
        num_classes = train_labels.max() + 1

    # 选择超参数
    logger.info("Using cuML for logistic regression")
    # best_stats, best_C = sweep_C_values(
    #     train_features=train_features,
    #     train_labels=train_labels,
    #     test_data_loader=finetune_data_loader,
    #     metric_type=metric_type,
    #     num_classes=num_classes,
    #     train_dtype=train_dtype,
    #     train_features_device=train_features_device,
    #     max_train_iters=max_train_iters,
    # )
    best_stats, best_C, all_res = sweep_C_values(
        train_features=train_features,
        train_labels=train_labels,
        test_data_loader=finetune_data_loader,
        metric_type=metric_type,
        num_classes=num_classes,
        train_dtype=train_dtype,
        train_features_device=train_features_device,
        max_train_iters=max_train_iters,
    )
    

    # 找到最佳参数时将数据拼起来再训一次logreg
    if not finetune_on_val:
        logger.info("Best parameter found, concatenating features")
        train_features = torch.cat((train_features, finetune_features))
        train_labels = torch.cat((train_labels, finetune_labels))
    logger.info("Training final model")
    logreg_metric = build_metric(metric_type, num_classes=num_classes)
    evals = train_and_evaluate(
        C=best_C,
        max_iter=max_train_iters,
        train_features=train_features,
        train_labels=train_labels,
        # logreg_metric=logreg_metric.clone(),
        logreg_metric=logreg_metric,
        test_data_loader=val_data_loader,
        eval_device=torch.cuda.current_device(),
        train_dtype=train_dtype,
        train_features_device=train_features_device,
    )

    best_stats = evals[1]["metrics"]
    best_stats["best_C"] = best_C
    tmp_results = evals[2]
    # 多个classifier的度量结果也一块保存
    best_stats["middle_results"] = all_res

    logger.info(f"Log regression evaluation done in {int(time.time() - start)}s")
    
    return best_stats, tmp_results, logreg_metric


def eval_log_regression_with_model(
        model,
        output_dir,
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        finetune_dataset_str=None,
        autocast_dtype=torch.float,
        finetune_on_val=False,
        metric_type=MetricType.MEAN_ACCURACY,
        train_dtype=torch.float64,
        train_features_device=_CPU_DEVICE,
        max_train_iters=DEFAULT_MAX_ITER,
        train_val_test_nums=[100, 100, 100],    # 此处输入数据大小划分
        batch_size=256,     # 此处控制提取特征时使用的batch size
        num_workers=0,
        transform=None,
        image_res=224
    ):
    cudnn.benchmark = True
    
    
    # 实例化transform
    # transform = make_classification_eval_transform(resize_size=224)
    ## modify决定输出图像尺寸，令其成为根据参数输入选择
    print(f"The image resolution is {image_res}")
    transform = transform or make_classification_eval_transform(resize_size=image_res,crop_size=image_res)
    target_transform = None

    
    # 实例化数据集
    # train_dataset = make_dataset(dataset_str=train_dataset_str, transform=transform, target_transform=target_transform)
    # val_dataset = make_dataset(dataset_str=val_dataset_str, transform=transform, target_transform=target_transform)
    ## 添加命令行输入的数据集大小
    train_dataset = make_dataset(dataset_str=train_dataset_str, 
                                 transform=transform, 
                                 target_transform=target_transform,
                                 train_num = train_val_test_nums[0],
                                 val_num = train_val_test_nums[1],
                                 test_num = train_val_test_nums[2],)
    val_dataset = make_dataset(dataset_str=val_dataset_str, 
                               transform=transform, 
                               target_transform=target_transform,
                               train_num = train_val_test_nums[0],
                               val_num = train_val_test_nums[1],
                               test_num = train_val_test_nums[2],)
    if finetune_dataset_str is not None:
        finetune_dataset = make_dataset(
            dataset_str=finetune_dataset_str, transform=transform, target_transform=target_transform
        )
    else:
        finetune_dataset = None
    
    # 训练logreg
    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        results_dict_logreg,tmp_results, logreg_metric = eval_log_regression(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            finetune_dataset=finetune_dataset,
            metric_type=metric_type,
            batch_size=batch_size,
            num_workers=num_workers,  # 5,
            finetune_on_val=finetune_on_val,
            train_dtype=train_dtype,
            train_features_device=train_features_device,
            max_train_iters=max_train_iters,
        )

    # results_dict = {
    #     "top-1": results_dict_logreg["top-1"].cpu().numpy() * 100.0,
    #     "top-5": results_dict_logreg.get("top-5", torch.tensor(0.0)).cpu().numpy() * 100.0,
    #     "best_C": results_dict_logreg["best_C"],
    # }
    # logger.info(
    #     "\n".join(
    #         [
    #             "Training of the supervised logistic regression on frozen features completed.\n"
    #             "Top-1 test accuracy: {acc:.1f}".format(acc=results_dict["top-1"]),
    #             "Top-5 test accuracy: {acc:.1f}".format(acc=results_dict["top-5"]),
    #             "obtained for C = {c:.6f}".format(c=results_dict["best_C"]),
    #         ]
    #     )
    # )
    
    # 记录结果
    def change_format(res_dict):
        a_results_dict = {}
        for topk in [1,5]:
            # acc
            for avg_mode in ['micro', 'macro', 'weighted']:
                a_top = res_dict[f"top-{topk}_acc_{avg_mode}"].item() * 100.0
                a_results_dict[f"Top {topk} acc {avg_mode}"] = a_top
            # 针对每个类的准确率记录
            per_class_acc = res_dict[f"top-{topk}_acc_none"].cpu().numpy() * 100.0
            a_results_dict[f"Top {topk} per-class-accuracy"] = per_class_acc.tolist()

            # f1
            for avg_mode in ['micro', 'macro', 'weighted']:
                a_top = res_dict[f"top-{topk}_f1_{avg_mode}"].item() * 100.0
                a_results_dict[f"Top {topk} f1 {avg_mode}"] = a_top
            # 针对每个类的准确率记录
            per_class_f1 = res_dict[f"top-{topk}_f1_none"].cpu().numpy() * 100.0
            a_results_dict[f"Top {topk} per-class-f1"] = per_class_f1.tolist()

            # auroc
            for avg_mode in ['macro', 'weighted']:
                a_top = res_dict[f"top-{topk}_auroc_{avg_mode}"].item() * 100.0
                a_results_dict[f"Top {topk} auroc {avg_mode}"] = a_top
            # 针对每个类的准确率记录
            per_class_f1 = res_dict[f"top-{topk}_auroc_none"].cpu().numpy() * 100.0
            a_results_dict[f"Top {topk} per-class-auroc"] = per_class_f1.tolist()

            # roc curve
            for avg_mode in ['macro']:
                tuple_list = res_dict[f"top-{topk}_roc_{avg_mode}"]
                a_results_dict[f"Top {topk} roc {avg_mode}"] = []
                fpr, tpr, thresholds = tuple_list
                a_results_dict[f"Top {topk} roc {avg_mode}"].append((fpr.cpu().numpy().tolist(),
                                                                    tpr.cpu().numpy().tolist(),
                                                                    thresholds.cpu().numpy().tolist()))

        return a_results_dict

    results_dict = change_format(results_dict_logreg)
    results_dict["best_C"] = results_dict_logreg["best_C"]
    results_dict["middle_results"] = []
    for a_tmp in results_dict_logreg["middle_results"]:
        a_results_dict = change_format(a_tmp)
        results_dict["middle_results"].append(a_results_dict)
    logger.info(
        "\n".join(
            [
                "Training of the supervised logistic regression on frozen features completed.\n"
                "Top-1 test micro accuracy: {acc:.1f}".format(acc=results_dict["Top 1 acc micro"]),
                "Top-1 test macro accuracy: {acc:.1f}".format(acc=results_dict["Top 1 acc macro"]),
                "Top-1 test micro f1: {f1:.1f}".format(f1=results_dict["Top 1 f1 micro"]),
                "Top-1 test macro f1: {f1:.1f}".format(f1=results_dict["Top 1 f1 macro"]),
                "Top-1 test macro auroc: {auroc:.1f}".format(auroc=results_dict["Top 1 auroc macro"]),
                "obtained for C = {c:.6f}".format(c=results_dict["best_C"]),
            ]
        )
    )
    
    
    # 保存到log dir路径中
    metrics_file_path = os.path.join(output_dir, "results_eval_logreg.json")
    with open(metrics_file_path, "w") as f:
        # f.write(json.dumps({f"Best_C {results_dict['best_C']} Top 1": results_dict["top-1"]}) + "\n")
        # f.write(json.dumps({f"Best_C {results_dict['best_C']} Top 5": results_dict["top-5"]}) + "\n")
        json.dump(results_dict, f, indent=4)
    
    
    # 保存预测结果
    pred_file_path = os.path.join(output_dir, "preds_eval_logreg.json")
    with open(pred_file_path, "w") as f:
        json.dump(tmp_results, f)
    
    
    # 保存roc curve
    for sub_key, sub_metric in logreg_metric.items():
        if '_roc_macro' in sub_key:
            roc_curve_file_path = os.path.join(output_dir, f"logs/roc_curve_best_{sub_key}.png")
            fig_, ax_ = sub_metric.plot(score=False)
            fig_.savefig(roc_curve_file_path)
    
    
    torch.distributed.barrier()
    return results_dict


def main(args):
    # 自动获得模型名称
    model_name = args.config_file.split('/')[-2]
    model_version_name = args.pretrained_weights.split('/')[-2]
    if model_name=='Resnet50':
        args.model_name = 'resnet50'
    elif model_name=='LVM_Resnet50':
        args.model_name = 'lvm_resnet50'
    elif model_name=='LVM_vitb16':
        args.model_name = 'lvm_vitb16'
    elif model_name=='Biomedclip':
        args.model_name = 'biomedclip'
        if model_version_name=='training_vitb16':
            args.model_version_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    elif model_name=='Clip':
        args.model_name = 'clip'
        if model_version_name=='training_vitb16':
            # args.model_version_name = 'openai/clip-vit-base-patch16'
            args.model_version_name = '/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint/hub/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3'
        elif model_version_name=='training_vitb32':
            # args.model_version_name = 'openai/clip-vit-base-patch32'
            args.model_version_name = '/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
        elif model_version_name=='training_vitl14':
            # args.model_version_name = 'openai/clip-vit-large-patch14'
            args.model_version_name = '/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
    elif model_name=='SAM':
        args.model_name = 'sam'
        if model_version_name=='training_vitb':
            args.model_version_name = 'facebook/sam-vit-base'
        elif model_version_name=='training_vith':
            args.model_version_name = 'facebook/sam-vit-huge'
        elif model_version_name=='training_vitl':
            args.model_version_name = 'facebook/sam-vit-large'
    elif model_name=='MedSAM':
        args.model_name = 'medsam'
        # args.model_version_name = 'wanglab/medsam-vit-base'
        args.model_version_name = '/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint/hub/models--wanglab--medsam-vit-base/snapshots/de8488bca37bb1d4fb190f612c516126d739ce3b'
    elif model_name=='SAM_Med2d':
        args.model_name = 'sammed2d'
    
    # 新增dinov3
    elif "Dinov3_Official_Model" in model_name:
        args.model_name = 'dinov3'
        if 'vitb' in model_name:
            args.model_version_name = 'vitb16'
            args.pretrained_weights = '/home/jovyan/dataset/Model_Outputs/dinov2/Dinov3_Official_Model_vitb16/eval/training_0001/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
        elif 'vitl' in model_name:
            args.model_version_name = 'vitl16'
            args.pretrained_weights = '/home/jovyan/dataset/Model_Outputs/dinov2/Dinov3_Official_Model_vitl16/eval/training_0001/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        elif 'vith' in model_name:
            args.model_version_name = 'vith16plus'
            args.pretrained_weights = '/home/jovyan/dataset/Model_Outputs/dinov2/Dinov3_Official_Model_vith16/eval/training_0001/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'
        elif 'vit7b' in model_name:
            args.model_version_name = 'vit7b16'
            args.pretrained_weights = '/home/jovyan/dataset/Model_Outputs/dinov2/Dinov3_Official_Model_vit7b/eval/training_0001/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth'
    
    else:
        args.model_name = 'dinov2'
    

    # load pretrained model
    if args.model_name=='resnet50':
        model, autocast_dtype = setup_and_build_model_resnet50(args)
        transform = None
    elif args.model_name=='lvm_resnet50':
        model, autocast_dtype = setup_and_build_model_lvm_resnet50(args)
        transform = None
    elif args.model_name=='lvm_vitb16':
        model, autocast_dtype = setup_and_build_model_lvm_vitb16(args)
        transform = None
    elif args.model_name=='biomedclip':
        model, autocast_dtype, a_preprocess = setup_and_build_model_biomedclip(args)
        transform = transforms.Compose([
                        BiomedClipTransform(a_preprocess)
                    ])
    elif args.model_name=='clip':
        model, autocast_dtype, a_preprocess = setup_and_build_model_clip(args)
        transform = transforms.Compose([
                        ClipTransform(a_preprocess)
                    ])
    elif args.model_name=='sam' or args.model_name=='medsam':
        model, autocast_dtype, a_preprocess = setup_and_build_model_sam(args)
        transform = transforms.Compose([
                        SAMTransform(a_preprocess)
                    ])
    elif args.model_name=='sammed2d':
        model, autocast_dtype = setup_and_build_model_sammed2d(args)
        transform = make_classification_eval_transform(resize_size=256, crop_size=256)
    
    # 新增dinov3
    elif args.model_name=='dinov3':
        model, autocast_dtype = setup_and_build_model_dinov3(args)
        transform = None
    
    else:
        model, autocast_dtype = setup_and_build_model(args)
        transform = None
    

    eval_log_regression_with_model(
        model=model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        finetune_dataset_str=args.finetune_dataset_str,
        autocast_dtype=autocast_dtype,
        finetune_on_val=args.finetune_on_val,
        metric_type=args.metric_type,
        train_dtype=as_torch_dtype(args.train_dtype),
        train_features_device=torch.device(args.train_features_device),
        max_train_iters=args.max_train_iters,
        train_val_test_nums=args.train_val_test_nums,   # 此处输入数据大小划分
        batch_size=args.batch_size,     # 此处控制提取特征时使用的batch size
        num_workers=0,
        transform=transform,
        image_res=args.image_res
    )
    
    return 0


if __name__ == "__main__":
    description = "DINOv2 logistic regression evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()

    # 判断输出路径是否存在，如果存在则先删除目录
    if os.path.exists(args.output_dir) and os.path.isdir(args.output_dir):
        print("Find existing dir!")
        if os.path.exists(os.path.join(args.output_dir,'results_eval_logreg.json')) and args.save_exist:
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
    
    sys.exit(main(args))