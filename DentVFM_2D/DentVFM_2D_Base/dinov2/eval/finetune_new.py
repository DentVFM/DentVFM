'''
评测全量微调(backbone+head)分类任务

1, 便捷切换backbone
2, head统一为Linear Layer
'''

import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data.transforms import make_classification_eval_transform, make_classification_train_transform
import dinov2.distributed as distributed
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model, setup_and_build_model_resnet50
from dinov2.eval.utils import ModelWithIntermediateLayers, evaluate
from dinov2.logging import MetricLogger
import matplotlib.pyplot as plt
import math
import copy


logger = logging.getLogger("dinov2")


# 加载参数
def get_args_parser(
        description: Optional[str] = None,
        parents: Optional[List[argparse.ArgumentParser]] = None,
        add_help: bool = True,
    ):
    parents = parents or []
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
    parser.add_argument(
        "--test-datasets",
        dest="test_dataset_strs",
        type=str,
        nargs="+",
        help="Test datasets, none to reuse the validation dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--epoch_length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--save_checkpoint_frequency",
        type=int,
        help="Number of epochs between two named checkpoint saves.",
    )
    parser.add_argument(
        "--eval_period_iterations",
        type=int,
        help="Number of iterations between two evaluations.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to grid search.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not resume from existing checkpoints",
    )
    parser.add_argument(
        "--val-metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Validation metric",
    )
    parser.add_argument(
        "--test-metric-types",
        type=MetricType,
        choices=list(MetricType),
        nargs="+",
        help="Evaluation metric",
    )
    parser.add_argument(
        "--classifier-fpath",
        type=str,
        help="Path to a file containing pretrained linear classifiers",
    )
    parser.add_argument(
        "--val-class-mapping-fpath",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        "--test-class-mapping-fpaths",
        nargs="+",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    # 新增命令行参数
    # 数据集划分大小
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
        help="backbone model name, can be 'resnet50', 'dinov2'",
    )
    # 是否删除已经存在的eval结果
    parser.add_argument(
        "--save_exist",
        action="store_true",
        help="do not remove existing dir",
    )
    # 图像分辨率，用于控制高分辨率
    parser.add_argument(
        "--image_res",
        type=int,
        help="Image resolution for training and evaluation.",
    )
    parser.set_defaults(
        model_name='dinov2',
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        test_dataset_strs=None,
        epochs=100,
        batch_size=32,  # 128
        num_workers=8,
        epoch_length=10,    # 1250
        save_checkpoint_frequency=20,
        eval_period_iterations=20,  # 1250
        learning_rates=[1e-2],  # [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]
        # learning_rates=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3],
        val_metric_type=MetricType.MEAN_ACCURACY,
        test_metric_types=None,
        classifier_fpath=None,
        val_class_mapping_fpath=None,
        test_class_mapping_fpaths=[None],
        train_val_test_nums=[100, 100, 100],
        image_res=224,
    )
    return parser


def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m


def _pad_and_collate(batch):
    maxlen = max(len(targets) for image, targets in batch)
    padded_batch = [
        (image, np.pad(targets, (0, maxlen - len(targets)), constant_values=-1)) for image, targets in batch
    ]
    return torch.utils.data.default_collate(padded_batch)

# 构造线性层的输入，取出encoder不同层后得到用于线性分类的特征
def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens 求所有patch特征的均值
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()

# 定义线性分类层,输入特征格式不同
# resnet50
class LinearClassifier_Resnet(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        return self.linear(x)
# dinov2
class LinearClassifier_Dino(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)
## 250905添加mlp
class LinearClassifier_Dino_o(nn.Module):
    """Three-layer MLP to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000, hidden_dim=2048):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes

        # 三层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

        # 初始化
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.mlp(output)
    


# 将dict封装为module形式，一次前向传播运行每个分类器
class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)

# 封装分类器，前向过程为分类，并对预测类别进行映射
class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, class_mapping=None):
        super().__init__()
        self.linear_classifier = linear_classifier
        self.register_buffer("class_mapping", None if class_mapping is None else torch.LongTensor(class_mapping))

    def forward(self, samples, targets):
        preds = self.linear_classifier(samples)
        return {
            "preds": preds[:, self.class_mapping] if self.class_mapping is not None else preds,
            "target": targets,
        }

# 缩放lr，此处lr是以256为标准的
def scale_lr(learning_rates, batch_size):
    return learning_rates * (batch_size * distributed.get_global_size()) / 256.0

# 配置线性分类器，每一种超参数的配置都会实例化一个分类器
def setup_linear_classifiers_dino(feature_model, sample_output, n_last_blocks_list, learning_rates, batch_size, num_classes=1000):
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in [False, True]:
            for _lr in learning_rates:
                lr = scale_lr(_lr, batch_size)
                # 计算fc层输入维度
                out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                linear_head = LinearClassifier_Dino(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                )
                linear_classifier = nn.Sequential(
                    copy.deepcopy(feature_model),
                    linear_head
                )

                linear_classifier.train()
                linear_classifier = linear_classifier.cuda()
                linear_classifiers_dict[
                    f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")
                ] = linear_classifier
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)    # 封装为整体
    if distributed.is_enabled():
        # linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers, find_unused_parameters=True)

    return linear_classifiers, optim_param_groups

## 250905修改，将深拷贝去除，适合单一配置
def setup_linear_classifiers_dino_n(feature_model, sample_output, n_last_blocks_list, learning_rates, batch_size, num_classes=1000, avgpool=[False]):
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in avgpool:
            for _lr in learning_rates:
                lr = scale_lr(_lr, batch_size)
                # 计算fc层输入维度
                out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                linear_head = LinearClassifier_Dino(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                )
                linear_classifier = nn.Sequential(
                    feature_model,
                    linear_head
                )
                feature_model.train()
                linear_classifier.train()
                linear_classifier = linear_classifier.cuda()
                linear_classifiers_dict[
                    f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")
                ] = linear_classifier
                # optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})
                optim_param_groups.append({"params": feature_model.parameters(), "lr": lr*0.05})
                optim_param_groups.append({"params": linear_head.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)    # 封装为整体
    if distributed.is_enabled():
        # linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers, find_unused_parameters=True)

    return linear_classifiers, optim_param_groups


def setup_linear_classifiers_resnet(feature_model, sample_output, n_last_blocks_list, learning_rates, batch_size, num_classes=1000):
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in [False]:
            for _lr in learning_rates:
                # lr = scale_lr(_lr, batch_size)
                lr = _lr
                out_dim = sample_output.shape[1]
                
                linear_head = LinearClassifier_Resnet(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                )

                linear_classifier = nn.Sequential(
                    copy.deepcopy(feature_model),
                    linear_head
                )
                
                linear_classifier.train()
                linear_classifier = linear_classifier.cuda()
                linear_classifiers_dict[
                    f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")
                ] = linear_classifier
                
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)    # 封装为整体
    if distributed.is_enabled():
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)

    return linear_classifiers, optim_param_groups


# 将模型评测指标转化为可序列化对象
def change_format(res_dict):
    a_results_dict = {}
    for topk in [1,5]:
        # acc
        for avg_mode in ['micro', 'macro', 'weighted']:
            a_top = res_dict[f"top-{topk}_acc_{avg_mode}"].item() * 100.0
            a_results_dict[f"top-{topk}_acc_{avg_mode}"] = a_top
        # 针对每个类的准确率记录
        per_class_acc = res_dict[f"top-{topk}_acc_none"].cpu().numpy() * 100.0
        a_results_dict[f"top-{topk}_acc_none"] = per_class_acc.tolist()

        # f1
        for avg_mode in ['micro', 'macro', 'weighted']:
            a_top = res_dict[f"top-{topk}_f1_{avg_mode}"].item() * 100.0
            a_results_dict[f"top-{topk}_f1_{avg_mode}"] = a_top
        # 针对每个类的准确率记录
        per_class_f1 = res_dict[f"top-{topk}_f1_none"].cpu().numpy() * 100.0
        a_results_dict[f"top-{topk}_f1_none"] = per_class_f1.tolist()

        # auroc
        for avg_mode in ['macro', 'weighted']:
            a_top = res_dict[f"top-{topk}_auroc_{avg_mode}"].item() * 100.0
            a_results_dict[f"top-{topk}_auroc_{avg_mode}"] = a_top
        # 针对每个类的准确率记录
        per_class_f1 = res_dict[f"top-{topk}_auroc_none"].cpu().numpy() * 100.0
        a_results_dict[f"top-{topk}_auroc_none"] = per_class_f1.tolist()

        # roc curve
        for avg_mode in ['macro']:
            tuple_list = res_dict[f"top-{topk}_roc_{avg_mode}"]
            a_results_dict[f"top-{topk}_roc_{avg_mode}"] = []
            fpr, tpr, thresholds = tuple_list
            a_results_dict[f"top-{topk}_roc_{avg_mode}"].append((fpr.cpu().numpy().tolist(),
                                                                tpr.cpu().numpy().tolist(),
                                                                thresholds.cpu().numpy().tolist()))
    return a_results_dict


@torch.no_grad()
def evaluate_linear_classifiers(
        feature_model,
        linear_classifiers,
        data_loader,
        metric_type,
        metrics_file_path,
        training_num_classes,
        iteration,
        prefixstring="",
        class_mapping=None,
        best_classifier_on_val=None,
    ):
    logger.info("running validation !")

    num_classes = len(class_mapping) if class_mapping is not None else training_num_classes
    metric = build_metric(metric_type, num_classes=num_classes)
    postprocessors = {k: LinearPostprocessor(v, class_mapping) for k, v in linear_classifiers.classifiers_dict.items()}
    metrics = {k: metric.clone() for k in linear_classifiers.classifiers_dict}
    
    # 将模型设置为eval模式
    for k, metric in metrics.items():
        postprocessors[k].eval()
    
    # 此处在evaluate中，对每个分类器进行评估
    _, results_dict_temp, tmp_results = evaluate(
        feature_model,
        data_loader,
        postprocessors,
        metrics,
        torch.cuda.current_device(),
    )

    # 将模型设置为train模式
    for k, metric in metrics.items():
        postprocessors[k].train()
    feature_model.train()

    logger.info("")
    results_dict = {}
    max_accuracy = 0
    best_classifier = ""
    for i, (classifier_string, metric) in enumerate(results_dict_temp.items()):
        current_eval_acc = metric["top-1_acc_micro"].item()
        logger.info(f"{prefixstring} -- Classifier: {classifier_string} * eval acc: {current_eval_acc}")
        
        # logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {metric}")
        # if (
        #     best_classifier_on_val is None and metric["top-1_acc_micro"].item() > max_accuracy
        # ) or classifier_string == best_classifier_on_val:
        #     max_accuracy = metric["top-1_acc_micro"].item()
        #     best_classifier = classifier_string

        if (
            best_classifier_on_val is None and metric["top-1_acc_micro"].item() > max_accuracy
        ) or classifier_string == best_classifier_on_val:
            max_accuracy = metric["top-1_acc_micro"].item()
            best_classifier = classifier_string


    # results_dict["best_classifier"] = {"name": best_classifier, "accuracy": max_accuracy}
    # 将所有的指标都保存!!
    results_dict["best_classifier"] = {"iter": iteration,
                                       "name": best_classifier, 
                                       "metrics": change_format(results_dict_temp[best_classifier]),
                                       "pred_results":tmp_results[best_classifier]}

    # logger.info(f"best classifier: {results_dict['best_classifier']}")
    # 只输出分类器名称
    logger.info(f"best classifier: {results_dict['best_classifier']['name']}")

    # if distributed.is_main_process():
    #     with open(metrics_file_path, "a") as f:
    #         f.write(f"iter: {iteration}\n")
    #         for k, v in results_dict.items():
    #             f.write(json.dumps({k: v}) + "\n")
    #         f.write("\n")

    return results_dict, results_dict_temp


def eval_linear(
        *,
        feature_model,
        linear_classifiers,
        train_data_loader,
        val_data_loader,
        metrics_file_path,
        optimizer,
        scheduler,
        output_dir,
        max_iter,
        checkpoint_period,  # In number of iter, creates a new file every period
        running_checkpoint_period,  # Period to update main checkpoint file
        eval_period,
        metric_type,
        training_num_classes,
        resume=True,
        classifier_fpath=None,
        val_class_mapping=None,
    ):
    checkpointer = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, checkpoint_period, max_iter=max_iter)
    iteration = start_iter
    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(delimiter="  ")
    header = "Training"
    
    # 此处记录每个分类器的训练loss以及eval的acc
    loss_records = {}
    loss_iter_records = []
    acc_records = {}
    acc_iter_records = []

    # 记录每次迭代的最好值，以及最终的最好值
    all_val_results = {}
    all_val_results['iter_best'] = []
    all_val_results['total_best'] = {"name":"",
                                     "best_metric":0}
    
    # 250905计算类别权重
    from collections import Counter
    def compute_class_weights(dataset, num_classes, device):
        # targets = []
        # if hasattr(dataset, 'target'):
        #     targets = dataset.target
        # elif hasattr(dataset, 'samples'):
        #     targets = [s[1] for s in dataset.samples]
        # else:
        #     raise ValueError("Dataset does not have targets or samples attribute")
        targets = dataset.get_targets()
        if targets is not None:
            targets = np.asarray(targets, dtype=np.int64)
        else:
            raise ValueError("Dataset does not have targets or samples attribute")
        print("total training number", targets.shape)
        # counts = Counter(targets)
        # total = sum(counts.values())
        # weights = [total / counts[i] for i in range(num_classes)]
        
        class_counts = np.bincount(targets, minlength=num_classes)
        class_counts = np.maximum(class_counts, 1)  # 避免除零
        weights = 1.0 / (class_counts ** 0.5)
        weights = weights / weights.sum() * num_classes
        return torch.tensor(weights, dtype=torch.float32, device=device)
    class_weights = compute_class_weights(train_data_loader.dataset, training_num_classes, device="cuda")
    print("class weights", class_weights)
    

    # 此处开始训练分类层
    for data, labels in metric_logger.log_every(
        train_data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        data = data.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        features = feature_model(data)
        outputs = linear_classifiers(features)  # 字典，存放每个参数配置下的classifier的output
        
        # 直接过分类器 debug
        # outputs = linear_classifiers(data)

        # losses = {f"loss_{k}": nn.CrossEntropyLoss()(v, labels) for k, v in outputs.items()}
        # 250905修改添加类别平衡
        losses = {f"loss_{k}": nn.CrossEntropyLoss(weight=class_weights)(v, labels) for k, v in outputs.items()}
        loss = sum(losses.values()) # 一块求个总loss，简便训练

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()
        scheduler.step()

        # log
        if iteration % 10 == 0:
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            # print("lr", optimizer.param_groups[0]["lr"])
            # 记录loss
            for k, v in losses.items():
                k = k.replace("loss_", "")
                loss_records[k] = loss_records.get(k, [])
                loss_records[k].append(v.item())
            loss_iter_records.append(iteration)


        if iteration - start_iter > 5:
            if iteration % running_checkpoint_period == 0:
                torch.cuda.synchronize()
                if distributed.is_main_process():
                    logger.info("Checkpointing running_checkpoint")
                    periodic_checkpointer.save("running_checkpoint_finetune_eval", iteration=iteration)
                torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        if eval_period > 0 and (iteration + 1) % eval_period == 0 and iteration != max_iter - 1:
            results_dict,results_dict_temp = evaluate_linear_classifiers(
                feature_model=feature_model,
                linear_classifiers=remove_ddp_wrapper(linear_classifiers),
                data_loader=val_data_loader,
                metrics_file_path=metrics_file_path,
                prefixstring=f"ITER: {iteration}",
                metric_type=metric_type,
                training_num_classes=training_num_classes,
                iteration=iteration,
                class_mapping=val_class_mapping,
            )
            torch.cuda.synchronize()

            # 记录当前iter的best cls结果
            all_val_results['iter_best'].append(results_dict['best_classifier'])
            ## 计算更新最佳iter
            if results_dict['best_classifier']['metrics']['top-1_acc_micro'] > all_val_results['total_best']['best_metric']:
                all_val_results['total_best']['best_metric'] = results_dict['best_classifier']['metrics']['top-1_acc_micro']
                all_val_results['total_best']['name'] = results_dict['best_classifier']['name']
                all_val_results['total_best']['iter'] = results_dict['best_classifier']['iter']
                all_val_results['total_best']['metrics'] = results_dict['best_classifier']['metrics']
                all_val_results['total_best']['pred_results'] = results_dict['best_classifier']['pred_results']

                # 保存最佳ckp
                torch.cuda.synchronize()
                if distributed.is_main_process():
                    logger.info("Checkpointing running_checkpoint")
                    periodic_checkpointer.save("best_checkpoint_eval", iteration=iteration)
                torch.cuda.synchronize()

            ## 将记录结果保存到json中
            if distributed.is_main_process():
                with open(metrics_file_path, "w") as f:
                    json.dump(all_val_results, f)
            
            # 记录acc
            for k, v in results_dict_temp.items():
                acc_records[k] = acc_records.get(k, [])
                # acc_records[k].append(v["top-1"].item())
                acc_records[k].append(v["top-1_acc_micro"].item())
            acc_iter_records.append(iteration)

            # 绘制loss曲线以及acc曲线
            n_plots = len(loss_records.keys())
            cols = math.ceil(math.sqrt(n_plots))
            rows = math.ceil(n_plots / cols)
            fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            if cols*rows > 1:
                axs = axs.flatten()
            else:
                axs = [axs]
            for i,k in enumerate(loss_records.keys()):
                ax = axs[i]
                # draw loss
                ax.plot(loss_iter_records, loss_records[k], color='r', label='Training Loss')
                ax.set_xlabel('Iterations')
                ax.set_ylabel('Training Loss', color='r')
                ax.tick_params(axis='y', labelcolor='r')
                # draw acc
                ax2 = ax.twinx()
                ax2.plot(acc_iter_records, acc_records[k], color='b', label='Val Acc')
                ax2.set_ylabel('Val Acc', color='b')
                ax2.tick_params(axis='y', labelcolor='b')
                
                ax.set_title(k)
            for j in range(n_plots, len(axs)):
                fig.delaxes(axs[j])
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'visualizations', 'loss_acc.png'))
            plt.close()

        iteration = iteration + 1


    val_results_dict,_ = evaluate_linear_classifiers(
        feature_model=feature_model,
        linear_classifiers=remove_ddp_wrapper(linear_classifiers),
        data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        metric_type=metric_type,
        training_num_classes=training_num_classes,
        iteration=iteration,
        class_mapping=val_class_mapping,
    )

    return val_results_dict, feature_model, linear_classifiers, iteration

# dataset构建可以放到外部
# def make_eval_data_loader(test_dataset_str, batch_size, num_workers, metric_type):
#     test_dataset = make_dataset(
#         dataset_str=test_dataset_str,
#         transform=make_classification_eval_transform(),
#     )
#     test_data_loader = make_data_loader(
#         dataset=test_dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         sampler_type=SamplerType.DISTRIBUTED,
#         drop_last=False,
#         shuffle=False,
#         persistent_workers=False,
#         collate_fn=_pad_and_collate if metric_type == MetricType.IMAGENET_REAL_ACCURACY else None,
#     )
#     return test_data_loader

def make_eval_data_loader(test_dataset, batch_size, num_workers, metric_type):
    test_data_loader = make_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
        collate_fn=_pad_and_collate if metric_type == MetricType.IMAGENET_REAL_ACCURACY else None,
    )
    return test_data_loader


def test_on_datasets(
        feature_model,
        linear_classifiers,
        test_dataset_strs,
        batch_size,
        num_workers,
        test_metric_types,
        metrics_file_path,
        training_num_classes,
        iteration,
        best_classifier_on_val,
        prefixstring="",
        test_class_mappings=[None],
    ):
    results_dict = {}
    for test_dataset_str, class_mapping, metric_type in zip(test_dataset_strs, test_class_mappings, test_metric_types):
        logger.info(f"Testing on {test_dataset_str}")
        test_data_loader = make_eval_data_loader(test_dataset_str, batch_size, num_workers, metric_type)
        dataset_results_dict,_ = evaluate_linear_classifiers(
            feature_model,
            remove_ddp_wrapper(linear_classifiers),
            test_data_loader,
            metric_type,
            metrics_file_path,
            training_num_classes,
            iteration,
            prefixstring="",
            class_mapping=class_mapping,
            best_classifier_on_val=best_classifier_on_val,
        )
        # results_dict[f"{test_dataset_str}_accuracy"] = 100.0 * dataset_results_dict["best_classifier"]["accuracy"]
        ## 此处保存测评结果
        results_dict[f"{test_dataset_str}_accuracy"] = dataset_results_dict['best_classifier']['metrics']['top-1_acc_micro']
    return results_dict


def run_eval_linear(
        model_name,
        model,
        output_dir,
        train_dataset_str,
        val_dataset_str,
        batch_size,
        epochs,
        epoch_length,
        num_workers,
        save_checkpoint_frequency,
        eval_period_iterations,
        learning_rates,
        autocast_dtype,
        test_dataset_strs=None,
        resume=True,
        classifier_fpath=None,
        val_class_mapping_fpath=None,
        test_class_mapping_fpaths=[None],
        val_metric_type=MetricType.MEAN_ACCURACY,
        test_metric_types=None,
        train_val_test_nums=[100, 100, 100],    # 此处输入数据大小划分
        img_res=224,
    ):
    seed = 0

    if test_dataset_strs is None:
        test_dataset_strs = [val_dataset_str]
    if test_metric_types is None:
        test_metric_types = [val_metric_type] * len(test_dataset_strs)
    else:
        assert len(test_metric_types) == len(test_dataset_strs)
    assert len(test_dataset_strs) == len(test_class_mapping_fpaths)

    print("Image resolution:", img_res)
    train_transform = make_classification_train_transform(crop_size=img_res)
    val_transform = make_classification_eval_transform(resize_size=img_res, crop_size=img_res)
    
    # train_dataset = make_dataset(
    #     dataset_str=train_dataset_str,
    #     transform=train_transform,
    # )
    
    # 添加自动设置数据集大小
    train_dataset = make_dataset(
        dataset_str=train_dataset_str,
        transform=train_transform,
        train_num = train_val_test_nums[0],
        val_num = train_val_test_nums[1],
        test_num = train_val_test_nums[2],
    )
    test_dataset = make_dataset(
        dataset_str=val_dataset_str,
        transform=val_transform,
        train_num = train_val_test_nums[0],
        val_num = train_val_test_nums[1],
        test_num = train_val_test_nums[2],
    )

    training_num_classes = len(torch.unique(torch.Tensor(train_dataset.get_targets().astype(int))))
    sampler_type = SamplerType.SHARDED_INFINITE
    # sampler_type = SamplerType.INFINITE
    
    # 构建分类模型
    feature_model = nn.Identity()
    feature_model.train()
    feature_model.cuda()
    # resnet50
    if model_name=='resnet50':
        n_last_blocks_list = [1]
        sample_output = model(train_dataset[0][0].unsqueeze(0).cuda())
        linear_classifiers, optim_param_groups = setup_linear_classifiers_resnet(
            model,
            sample_output,
            n_last_blocks_list,
            learning_rates,
            batch_size,
            training_num_classes,
        )
    else:
        # n_last_blocks_list = [1, 4]
        n_last_blocks_list = [1]
        n_last_blocks = max(n_last_blocks_list)
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
        # feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)
        model_1 = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)   # 可以从预训练模型中提取多层特征的封装
        # sample_output = feature_model(train_dataset[0][0].unsqueeze(0).cuda())
        sample_output = model_1(train_dataset[0][0].unsqueeze(0).cuda())
        linear_classifiers, optim_param_groups = setup_linear_classifiers_dino(
            model_1,
            sample_output,
            n_last_blocks_list,
            learning_rates,
            batch_size,
            training_num_classes,
        )

    optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
    max_iter = epochs * epoch_length
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
    checkpointer = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1
    
    
    train_data_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        sampler_type=sampler_type,
        sampler_advance=start_iter,
        drop_last=True,
        persistent_workers=True,
    )
    # val_data_loader = make_eval_data_loader(val_dataset_str, batch_size, num_workers, val_metric_type)
    val_data_loader = make_eval_data_loader(test_dataset, batch_size, num_workers, val_metric_type)

    checkpoint_period = save_checkpoint_frequency * epoch_length

    if val_class_mapping_fpath is not None:
        logger.info(f"Using class mapping from {val_class_mapping_fpath}")
        val_class_mapping = np.load(val_class_mapping_fpath)
    else:
        val_class_mapping = None

    test_class_mappings = []
    for class_mapping_fpath in test_class_mapping_fpaths:
        if class_mapping_fpath is not None and class_mapping_fpath != "None":
            logger.info(f"Using class mapping from {class_mapping_fpath}")
            class_mapping = np.load(class_mapping_fpath)
        else:
            class_mapping = None
        test_class_mappings.append(class_mapping)

    metrics_file_path = os.path.join(output_dir, "results_eval_finetune.json")
    val_results_dict, feature_model, linear_classifiers, iteration = eval_linear(
        feature_model=feature_model,
        linear_classifiers=linear_classifiers,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        max_iter=max_iter,
        checkpoint_period=checkpoint_period,
        running_checkpoint_period=epoch_length,
        eval_period=eval_period_iterations,
        metric_type=val_metric_type,
        training_num_classes=training_num_classes,
        resume=resume,
        val_class_mapping=val_class_mapping,
        classifier_fpath=classifier_fpath,
    )
    results_dict = {}

    # if len(test_dataset_strs) > 1 or test_dataset_strs[0] != val_dataset_str:
        # results_dict = test_on_datasets(
        #     feature_model,
        #     linear_classifiers,
        #     test_dataset_strs,
        #     batch_size,
        #     0,  # num_workers,
        #     test_metric_types,
        #     metrics_file_path,
        #     training_num_classes,
        #     iteration,
        #     val_results_dict["best_classifier"]["name"],
        #     prefixstring="",
        #     test_class_mappings=test_class_mappings,
        # )
    
    results_dict["best_classifier"] = val_results_dict["best_classifier"]["name"]
    # results_dict[f"{val_dataset_str}_accuracy"] = 100.0 * val_results_dict["best_classifier"]["accuracy"]
    ## 此处保存测评结果
    results_dict[f"{val_dataset_str}_accuracy"] = val_results_dict['best_classifier']['metrics']['top-1_acc_micro']
    logger.info("Test Results Dict " + str(results_dict))

    return results_dict






def main(args):
    # 自动获得模型名称
    model_name = args.config_file.split('/')[-2]
    model_version_name = args.pretrained_weights.split('/')[-2]

    if model_name=='Resnet50':
        args.model_name = 'resnet50'
    else:
        args.model_name = 'dinov2'
        
    # load pretrained model
    if args.model_name=='resnet50':
        model, autocast_dtype = setup_and_build_model_resnet50(args)
    else:
        model, autocast_dtype = setup_and_build_model(args)
    
    run_eval_linear(
        model_name=args.model_name,
        model=model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        test_dataset_strs=args.test_dataset_strs,
        batch_size=args.batch_size,
        epochs=args.epochs,
        epoch_length=args.epoch_length,
        num_workers=args.num_workers,
        save_checkpoint_frequency=args.save_checkpoint_frequency,
        eval_period_iterations=args.eval_period_iterations,
        learning_rates=args.learning_rates,
        autocast_dtype=autocast_dtype,
        resume=not args.no_resume,
        classifier_fpath=args.classifier_fpath,
        val_metric_type=args.val_metric_type,
        test_metric_types=args.test_metric_types,
        val_class_mapping_fpath=args.val_class_mapping_fpath,
        test_class_mapping_fpaths=args.test_class_mapping_fpaths,
        train_val_test_nums=args.train_val_test_nums,   # 此处输入数据大小划分
        img_res=args.image_res,
    )
    return 0





if __name__ == "__main__":
    description = "Full Finetune Evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()

    # 判断输出路径是否存在，如果存在则先删除目录
    if os.path.exists(args.output_dir) and os.path.isdir(args.output_dir):
        print("Find existing dir!")
        if os.path.exists(os.path.join(args.output_dir,'results_eval_finetune.json')) and args.save_exist:
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
    
    # 创建保存训练可视化结果的目录
    os.makedirs(os.path.join(args.output_dir, 'visualizations'),exist_ok=True)

    sys.exit(main(args))