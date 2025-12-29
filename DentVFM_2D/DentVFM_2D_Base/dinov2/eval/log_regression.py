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
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import evaluate, extract_features
from dinov2.utils.dtype import as_torch_dtype


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

    # 此处添加命令行输入的数据集划分大小
    parser.add_argument(
        "--train_val_test_nums",
        nargs="+",
        type=int,
        help="dataset split number",
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
    # 是否删除已经存在的eval结果
    parser.add_argument(
        "--save_exist",
        action="store_true",
        help="do not remove existing dir",
    )
    parser.set_defaults(
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
    )
    return parser


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

        best_stats, which_epoch = metric_tracker.best_metric(return_step=True)
        best_stats_100 = {k: 100.0 * v for k, v in best_stats.items()}
        if which_epoch["top-1"] == i:
            best_C = C
    logger.info(f"Sweep best {best_stats_100}, best C = {best_C:.6f}")

    return best_stats, best_C


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
    best_stats, best_C = sweep_C_values(
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
        logreg_metric=logreg_metric.clone(),
        test_data_loader=val_data_loader,
        eval_device=torch.cuda.current_device(),
        train_dtype=train_dtype,
        train_features_device=train_features_device,
    )

    best_stats = evals[1]["metrics"]
    best_stats["best_C"] = best_C
    tmp_results = evals[2]

    logger.info(f"Log regression evaluation done in {int(time.time() - start)}s")
    
    return best_stats, tmp_results


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
    ):
    cudnn.benchmark = True

    transform = make_classification_eval_transform(resize_size=224)
    target_transform = None

    # train_dataset = make_dataset(dataset_str=train_dataset_str, transform=transform, target_transform=target_transform)
    # val_dataset = make_dataset(dataset_str=val_dataset_str, transform=transform, target_transform=target_transform)
    
    # 添加命令行输入的数据集大小
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

    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        results_dict_logreg,tmp_results = eval_log_regression(
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

    results_dict = {
        "top-1": results_dict_logreg["top-1"].cpu().numpy() * 100.0,
        "top-5": results_dict_logreg.get("top-5", torch.tensor(0.0)).cpu().numpy() * 100.0,
        "best_C": results_dict_logreg["best_C"],
    }
    logger.info(
        "\n".join(
            [
                "Training of the supervised logistic regression on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=results_dict["top-1"]),
                "Top-5 test accuracy: {acc:.1f}".format(acc=results_dict["top-5"]),
                "obtained for C = {c:.6f}".format(c=results_dict["best_C"]),
            ]
        )
    )

    # 保存到log dir路径中
    metrics_file_path = os.path.join(output_dir, "results_eval_logreg.json")
    with open(metrics_file_path, "w") as f:
        f.write(json.dumps({f"Best_C {results_dict['best_C']} Top 1": results_dict["top-1"]}) + "\n")
        f.write(json.dumps({f"Best_C {results_dict['best_C']} Top 5": results_dict["top-5"]}) + "\n")
    
    # 保存预测结果
    pred_file_path = os.path.join(output_dir, "preds_eval_logreg.json")
    with open(pred_file_path, "w") as f:
        json.dump(tmp_results, f)

    torch.distributed.barrier()
    return results_dict


def main(args):
    # 加载训练好的模型
    model, autocast_dtype = setup_and_build_model(args)
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