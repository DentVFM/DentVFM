# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional
import glob
import shutil
import torch
from torch.nn.functional import one_hot, softmax
import dinov2.distributed as distributed
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.eval.metrics import AccuracyAveraging, build_topk_accuracy_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model,setup_and_build_model_resnet50,setup_and_build_model_biomedclip, setup_and_build_model_clip, setup_and_build_model_sam, setup_and_build_model_lvm_resnet50, setup_and_build_model_lvm_vitb16, setup_and_build_model_sammed2d
from dinov2.eval.utils import ModelWithNormalize, evaluate, extract_features
from torchvision import transforms
import numpy as np

logger = logging.getLogger("dinov2")

# 设定knn相关的参数
def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
    ):
    '''
    设定knn评测使用的参数
    
    其他评测使用的通用参数参数继承自dinov2.eval.setup的基础参数
    '''
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
        "--nb_knn",
        nargs="+",
        type=int,
        help="Number of NN to use. 20 is usually working the best.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature used in the voting coefficient",
    )
    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the train features on cpu, slower"
        "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--n-per-class-list",
        nargs="+",
        type=int,
        help="Number to take per class",
    )
    parser.add_argument(
        "--n-tries",
        type=int,
        help="Number of tries",
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
    parser.set_defaults(
        model_name='dinov2',
        model_version_name='training_vitb16',
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        nb_knn=[10, 20, 100, 200],
        temperature=0.07,
        batch_size=256,
        n_per_class_list=[-1],
        n_tries=1,
        train_val_test_nums=[100, 100, 100],
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

class KnnModule(torch.nn.Module):
    """
    Gets knn of test features from all processes on a chunk of the train features

    Each rank gets a chunk of the train features as well as a chunk of the test features.
    In `compute_neighbors`, for each rank one after the other, its chunk of test features
    is sent to all devices, partial knns are computed with each chunk of train features
    then collated back on the original device.
    """

    def __init__(self, train_features, train_labels, nb_knn, T, device, num_classes=1000):
        super().__init__()

        self.global_rank = distributed.get_global_rank()
        self.global_size = distributed.get_global_size()

        self.device = device
        self.train_features_rank_T = train_features.chunk(self.global_size)[self.global_rank].T.to(self.device)
        self.candidates = train_labels.chunk(self.global_size)[self.global_rank].view(1, -1).to(self.device)

        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.T = T
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(self, similarity, train_labels):
        topk_sims, indices = similarity.topk(self.max_k, largest=True, sorted=True)
        neighbors_labels = torch.gather(train_labels, 1, indices)   # 取出topk的train样本的label
        return topk_sims, neighbors_labels

    def _similarity_for_rank(self, features_rank, source_rank):
        # Send the features from `source_rank` to all ranks
        broadcast_shape = torch.tensor(features_rank.shape).to(self.device)
        torch.distributed.broadcast(broadcast_shape, source_rank)

        broadcasted = features_rank
        if self.global_rank != source_rank:
            broadcasted = torch.zeros(*broadcast_shape, dtype=features_rank.dtype, device=self.device)
        torch.distributed.broadcast(broadcasted, source_rank)   # 将val数据广播到不同node上

        # Compute the neighbors for `source_rank` among `train_features_rank_T`
        similarity_rank = torch.mm(broadcasted, self.train_features_rank_T) # 计算相似性是对每个chunck计算相似性
        candidate_labels = self.candidates.expand(len(similarity_rank), -1) # (n,m)
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def _gather_all_knn_for_rank(self, topk_sims, neighbors_labels, target_rank):
        # Gather all neighbors for `target_rank`
        topk_sims_rank = retrieved_rank = None
        if self.global_rank == target_rank:
            topk_sims_rank = [torch.zeros_like(topk_sims) for _ in range(self.global_size)]
            retrieved_rank = [torch.zeros_like(neighbors_labels) for _ in range(self.global_size)]

        torch.distributed.gather(topk_sims, topk_sims_rank, dst=target_rank)
        torch.distributed.gather(neighbors_labels, retrieved_rank, dst=target_rank)

        if self.global_rank == target_rank:
            # Perform a second top-k on the k * global_size retrieved neighbors
            topk_sims_rank = torch.cat(topk_sims_rank, dim=1)   # 将每个rank上的topk拼接
            retrieved_rank = torch.cat(retrieved_rank, dim=1)
            results = self._get_knn_sims_and_labels(topk_sims_rank, retrieved_rank) # 从合并后的结果中找到topk
            return results
        return None

    def compute_neighbors(self, features_rank):
        for rank in range(self.global_size):
            topk_sims, neighbors_labels = self._similarity_for_rank(features_rank, rank)    # 从不同rank计算topk相似度
            results = self._gather_all_knn_for_rank(topk_sims, neighbors_labels, rank)      # 聚合不同rank的结果
            if results is not None:
                topk_sims_rank, neighbors_labels_rank = results
        return topk_sims_rank, neighbors_labels_rank

    def forward(self, features_rank):
        """
        Compute the results on all values of `self.nb_knn` neighbors from the full `self.max_k`
        """
        assert all(k <= self.max_k for k in self.nb_knn)

        topk_sims, neighbors_labels = self.compute_neighbors(features_rank) # 计算topk相似度以及对应label
        batch_size = neighbors_labels.shape[0]
        topk_sims_transform = softmax(topk_sims / self.T, 1)    # (n,topk)归一化每个样本的相似度
        matmul = torch.mul(
            one_hot(neighbors_labels, num_classes=self.num_classes),
            topk_sims_transform.view(batch_size, -1, 1),
        )   # 把one-hot中的1变为相似度值
        probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in self.nb_knn} # 统计每种k取值下每个类别的概率大小
        return probas_for_k


class DictKeysModule(torch.nn.Module):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def forward(self, features_dict, targets):  # 输入是预测的结果和target，预测的结果是dict形式
        for k in self.keys:                     # 目的是从features_dict中取出预测结果，因为features_dict是嵌套形式，需要逐层打开
            features_dict = features_dict[k]
        return {"preds": features_dict, "target": targets}


def create_module_dict(*, module, n_per_class_list, n_tries, nb_knn, train_features, train_labels):
    modules = {}
    mapping = create_class_indices_mapping(train_labels)
    for npc in n_per_class_list:
        if npc < 0:  # Only one try needed when using the full data
            full_module = module(
                train_features=train_features,
                train_labels=train_labels,
                nb_knn=nb_knn,
            )
            modules["full"] = ModuleDictWithForward({"1": full_module})
            continue
        all_tries = {}
        for t in range(n_tries):    # 因为有随机性的存在，此处尝试多次，上面full的话只需要尝试一次即可
            final_indices = filter_train(mapping, npc, seed=t)  # 每个class的train样本中随机取npc个
            k_list = list(set(nb_knn + [npc]))
            k_list = sorted([el for el in k_list if el <= npc]) # 因为每个类别下采样了，所以knn的k的个数需要少于npc，此处重新设置
            all_tries[str(t)] = module(
                train_features=train_features[final_indices],
                train_labels=train_labels[final_indices],
                nb_knn=k_list,
            )
        modules[f"{npc} per class"] = ModuleDictWithForward(all_tries)

    return ModuleDictWithForward(modules)


def filter_train(mapping, n_per_class, seed):
    torch.manual_seed(seed)
    final_indices = []
    for k in mapping.keys():
        index = torch.randperm(len(mapping[k]))[:n_per_class]   # 每个类别的train样本中随机取n_per_class个
        final_indices.append(mapping[k][index])
    return torch.cat(final_indices).squeeze()


def create_class_indices_mapping(labels):
    unique_labels, inverse = torch.unique(labels, return_inverse=True)
    mapping = {unique_labels[i]: (inverse == i).nonzero() for i in range(len(unique_labels))}
    return mapping


class ModuleDictWithForward(torch.nn.ModuleDict):
    def forward(self, *args, **kwargs): # 相当于对dict中的每个模型都执行forward
        return {k: module(*args, **kwargs) for k, module in self._modules.items()}


def eval_knn(
        model,
        train_dataset,
        val_dataset,
        accuracy_averaging,
        nb_knn,
        temperature,
        batch_size,
        num_workers,
        gather_on_cpu,
        n_per_class_list=[-1],
        n_tries=1,
    ):
    model = ModelWithNormalize(model)

    logger.info("Extracting features for train set...")
    train_features, train_labels = extract_features(
        model, train_dataset, batch_size, num_workers, gather_on_cpu=gather_on_cpu
    )
    logger.info(f"Train features created, shape {train_features.shape}.")

    val_dataloader = make_data_loader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,  # 此处不进行shuffle
        persistent_workers=True,
    )
    num_classes = train_labels.max() + 1
    metric_collection = build_topk_accuracy_metric(accuracy_averaging, num_classes=num_classes) # 计算评价指标的对象，会评价topk的情况

    device = torch.cuda.current_device()
    partial_module = partial(KnnModule, T=temperature, device=device, num_classes=num_classes)
    knn_module_dict = create_module_dict(
        module=partial_module,
        n_per_class_list=n_per_class_list,
        n_tries=n_tries,
        nb_knn=nb_knn,
        train_features=train_features,
        train_labels=train_labels,
    )
    postprocessors, metrics = {}, {}
    for n_per_class, knn_module in knn_module_dict.items(): # 对每个class取多少样本数进行遍历
        for t, knn_try in knn_module.items():   # 对每种取数情况下多次try进行遍历
            postprocessors = {
                **postprocessors,
                **{(n_per_class, t, k): DictKeysModule([n_per_class, t, k]) for k in knn_try.nb_knn},   # 此处还需要遍历每个k的取值
            }
            metrics = {**metrics, **{(n_per_class, t, k): metric_collection.clone() for k in knn_try.nb_knn}}
    model_with_knn = torch.nn.Sequential(model, knn_module_dict)    # 将模型和计算预测结果连接

    # ============ evaluation ... ============
    logger.info("Start the k-NN classification.")
    _, results_dict, tmp_results= evaluate(model_with_knn, val_dataloader, postprocessors, metrics, device)

    # Averaging the results over the n tries for each value of n_per_class
    for n_per_class, knn_module in knn_module_dict.items():
        first_try = list(knn_module.keys())[0]
        k_list = knn_module[first_try].nb_knn
        for k in k_list:
            keys = results_dict[(n_per_class, first_try, k)].keys()  # keys are e.g. `top-1` and `top-5`
            
            # 此处对t，即所有tries做了一次平均
            # 即每个n_per_class+k会对应一个多次测试的平均值
            # results_dict[(n_per_class, k)] = {
            #     key: torch.mean(torch.stack([results_dict[(n_per_class, t, k)][key] for t in knn_module.keys()]))
            #     for key in keys
            # }
            
            # 针对不同的keys，需要使用不同的合并方式
            ## 简单求平均
            keys_1 = [key for key in keys if any(sub in key for sub in ["acc_micro","acc_macro","acc_weighted","f1_micro",
                                                                        "f1_macro","f1_weighted","auroc_macro","auroc_weighted"])]
            results_dict[(n_per_class, k)] = {
                key: torch.mean(torch.stack([results_dict[(n_per_class, t, k)][key] for t in knn_module.keys()]))
                for key in keys_1
            }
            ## tensor对第一维度求平均
            keys_2 = [key for key in keys if any(sub in key for sub in ["acc_none","f1_none","auroc_none"])]
            results_dict[(n_per_class, k)].update(
                {
                    key: torch.mean(torch.stack([results_dict[(n_per_class, t, k)][key] for t in knn_module.keys()]), dim=0)
                    for key in keys_2
                }
            )
            ## roc包含tuple输出，不能求平均直接保存其tuple list [(tpr, fpr, thresholds),...,]
            keys_3 = [key for key in keys if any(sub in key for sub in ["_roc_macro"])]
            results_dict[(n_per_class, k)].update(
                {
                    key: [results_dict[(n_per_class, t, k)][key] for t in knn_module.keys()]
                    for key in keys_3
                }
            )
            
            for t in knn_module.keys():
                del results_dict[(n_per_class, t, k)]

    return results_dict, tmp_results, metrics


def eval_knn_with_model(
    model,
    output_dir,
    train_dataset_str="ImageNet:split=TRAIN",
    val_dataset_str="ImageNet:split=VAL",
    nb_knn=(10, 20, 100, 200),
    temperature=0.07,
    autocast_dtype=torch.float,
    accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
    transform=None,
    gather_on_cpu=False,
    batch_size=256,
    num_workers=5,
    n_per_class_list=[-1],
    n_tries=1,
    train_val_test_nums=[100, 100, 100],    # 此处输入数据大小划分
    ):
    transform = transform or make_classification_eval_transform()  # 定义图片需要做的变换

    # train_dataset = make_dataset(
    #     dataset_str=train_dataset_str,
    #     transform=transform,
    # )
    # val_dataset = make_dataset(
    #     dataset_str=val_dataset_str,
    #     transform=transform,
    # )

    # 添加命令行输入的数据集大小
    train_dataset = make_dataset(
        dataset_str=train_dataset_str,
        transform=transform,
        train_num = train_val_test_nums[0],
        val_num = train_val_test_nums[1],
        test_num = train_val_test_nums[2],
    )
    val_dataset = make_dataset(
        dataset_str=val_dataset_str,
        transform=transform,
        train_num = train_val_test_nums[0],
        val_num = train_val_test_nums[1],
        test_num = train_val_test_nums[2],
    )

    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        results_dict_knn,tmp_results, metrics = eval_knn(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            accuracy_averaging=accuracy_averaging,
            nb_knn=nb_knn,
            temperature=temperature,
            batch_size=batch_size,
            num_workers=num_workers,
            gather_on_cpu=gather_on_cpu,
            n_per_class_list=n_per_class_list,
            n_tries=n_tries,
        )
    
    # 整理指标结果
    # results_dict = {}
    # if distributed.is_main_process():
    #     for knn_ in results_dict_knn.keys():
    #         top1 = results_dict_knn[knn_]["top-1"].item() * 100.0
    #         top5 = results_dict_knn[knn_]["top-5"].item() * 100.0
    #         results_dict[f"{knn_} Top 1"] = top1
    #         results_dict[f"{knn_} Top 5"] = top5
    #         logger.info(f"{knn_} classifier result: Top1: {top1:.2f} Top5: {top5:.2f}")
    
    # 整理指标结果, 添加了更多的评测指标 !!
    results_dict = {}
    if distributed.is_main_process():
        for knn_ in results_dict_knn.keys():
            
            log_str = f"{knn_} classifier result: "
            for topk in [1,5]:
                # acc
                for avg_mode in ['micro', 'macro', 'weighted']:
                    a_top = results_dict_knn[knn_][f"top-{topk}_acc_{avg_mode}"].item() * 100.0
                    results_dict[f"{knn_} Top {topk} acc {avg_mode}"] = a_top
                    log_str += f"Top {topk} acc {avg_mode}: {a_top:.2f} "
                # 针对每个类的准确率记录
                per_class_acc = results_dict_knn[knn_][f"top-{topk}_acc_none"].cpu().numpy() * 100.0
                results_dict[f"{knn_} Top {topk} per-class-accuracy"] = per_class_acc.tolist()

                # f1
                for avg_mode in ['micro', 'macro', 'weighted']:
                    a_top = results_dict_knn[knn_][f"top-{topk}_f1_{avg_mode}"].item() * 100.0
                    results_dict[f"{knn_} Top {topk} f1 {avg_mode}"] = a_top
                    log_str += f"Top {topk} f1 {avg_mode}: {a_top:.2f} "
                # 针对每个类的准确率记录
                per_class_f1 = results_dict_knn[knn_][f"top-{topk}_f1_none"].cpu().numpy() * 100.0
                results_dict[f"{knn_} Top {topk} per-class-f1"] = per_class_f1.tolist()

                # auroc
                for avg_mode in ['macro', 'weighted']:
                    a_top = results_dict_knn[knn_][f"top-{topk}_auroc_{avg_mode}"].item() * 100.0
                    results_dict[f"{knn_} Top {topk} auroc {avg_mode}"] = a_top
                    log_str += f"Top {topk} auroc {avg_mode}: {a_top:.2f} "
                # 针对每个类的准确率记录
                per_class_f1 = results_dict_knn[knn_][f"top-{topk}_auroc_none"].cpu().numpy() * 100.0
                results_dict[f"{knn_} Top {topk} per-class-auroc"] = per_class_f1.tolist()

                # roc curve
                for avg_mode in ['macro']:
                    tuple_list = results_dict_knn[knn_][f"top-{topk}_roc_{avg_mode}"]
                    results_dict[f"{knn_} Top {topk} roc {avg_mode}"] = []
                    for a_tuple in tuple_list:
                        fpr, tpr, thresholds = a_tuple
                        results_dict[f"{knn_} Top {topk} roc {avg_mode}"].append((fpr.cpu().numpy().tolist(),
                                                                                  tpr.cpu().numpy().tolist(),
                                                                                  thresholds.cpu().numpy().tolist()))
                                  
            logger.info(log_str)
    
    # 保存knn评价结果
    metrics_file_path = os.path.join(output_dir, "results_eval_knn.json")
    with open(metrics_file_path, "w") as f:
        # for k, v in results_dict.items():
        #     f.write(json.dumps({k: v}) + "\n")
        json.dump(results_dict, f, indent=4)
    
    # 保存预测结果
    pred_file_path = os.path.join(output_dir, "preds_eval_knn.json")
    with open(pred_file_path, "w") as f:
        json.dump(tmp_results, f)
    
    # 保存roc curve
    for metric_key, a_metric in metrics.items():
        for sub_key, sub_metric in a_metric.items():
            if '_roc_macro' in sub_key:
                roc_curve_file_path = os.path.join(output_dir, f"logs/roc_curve_{metric_key}_{sub_key}.png")
                fig_, ax_ = sub_metric.plot(score=False)
                fig_.savefig(roc_curve_file_path)
    
    
    if distributed.is_enabled():
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
    else:
        model, autocast_dtype = setup_and_build_model(args)
        transform = None

    # 开始评价模型
    eval_knn_with_model(
        model=model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        nb_knn=args.nb_knn,
        temperature=args.temperature,
        autocast_dtype=autocast_dtype,
        accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
        transform=transform,        ## debug None
        gather_on_cpu=args.gather_on_cpu,
        batch_size=args.batch_size, # 此处batch size为命令行输入参数
        num_workers=5,
        n_per_class_list=args.n_per_class_list,
        n_tries=args.n_tries,
        train_val_test_nums=args.train_val_test_nums,   # 此处输入数据大小划分
    )
    return 0


if __name__ == "__main__":
    description = "DINOv2 k-NN evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    
    # 判断输出路径是否存在，如果存在则先删除目录
    if os.path.exists(args.output_dir) and os.path.isdir(args.output_dir):
        print("Find existing dir!")
        if os.path.exists(os.path.join(args.output_dir,'results_eval_knn.json')) and args.save_exist:
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