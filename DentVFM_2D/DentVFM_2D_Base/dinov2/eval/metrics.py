# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from enum import Enum
import logging
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassROC
from torchmetrics.utilities.data import dim_zero_cat, select_topk


logger = logging.getLogger("dinov2")


class MetricType(Enum):
    MEAN_ACCURACY = "mean_accuracy"
    MEAN_PER_CLASS_ACCURACY = "mean_per_class_accuracy"
    PER_CLASS_ACCURACY = "per_class_accuracy"
    IMAGENET_REAL_ACCURACY = "imagenet_real_accuracy"

    @property
    def accuracy_averaging(self):
        return getattr(AccuracyAveraging, self.name, None)

    def __str__(self):
        return self.value


class AccuracyAveraging(Enum):
    MEAN_ACCURACY = "micro"
    MEAN_PER_CLASS_ACCURACY = "macro"
    PER_CLASS_ACCURACY = "none"

    def __str__(self):
        return self.value


def build_metric(metric_type: MetricType, *, num_classes: int, ks: Optional[tuple] = None):
    if metric_type.accuracy_averaging is not None:
        return build_topk_accuracy_metric(
            average_type=metric_type.accuracy_averaging,
            num_classes=num_classes,
            ks=(1, 5) if ks is None else ks,
        )
    elif metric_type == MetricType.IMAGENET_REAL_ACCURACY:
        return build_topk_imagenet_real_accuracy_metric(
            num_classes=num_classes,
            ks=(1, 5) if ks is None else ks,
        )

    raise ValueError(f"Unknown metric type {metric_type}")


# def build_topk_accuracy_metric(average_type: AccuracyAveraging, num_classes: int, ks: tuple = (1, 5)):
#     # metrics: Dict[str, Metric] = {
#     #     f"top-{k}": MulticlassAccuracy(top_k=k, num_classes=int(num_classes), average=average_type.value) for k in ks
#     # }
#     metrics: Dict[str, Metric] = {
#         f"top-{k}": MulticlassAccuracy(top_k=1, num_classes=int(num_classes), average=average_type.value) for k in ks
#     }
#     return MetricCollection(metrics)

# 修改为添加weighted acc, f1 score, auroc, roc曲线
def build_topk_accuracy_metric(average_type: AccuracyAveraging, num_classes: int, ks: tuple = (1, 5)):
    # 计算micro非weighted
    metrics: Dict[str, Metric] = {
        f"top-{k}_acc_micro": MulticlassAccuracy(top_k=1, num_classes=int(num_classes), average=average_type.value) for k in ks
    }
    # 计算macro非weighted
    metrics.update(
        {
            f"top-{k}_acc_macro": MulticlassAccuracy(top_k=1, num_classes=int(num_classes), average="macro") for k in ks
        }
    )
    # 计算weighted
    metrics.update(
        {
            f"top-{k}_acc_weighted": MulticlassAccuracy(top_k=1, num_classes=int(num_classes), average="weighted") for k in ks
        }
    )
    # 计算并保留每个类别的准确率
    metrics.update(
        {
            f"top-{k}_acc_none": MulticlassAccuracy(top_k=1, num_classes=int(num_classes), average="none") for k in ks
        }
    )
    # 计算f1 score
    metrics.update(
        {
            f"top-{k}_f1_micro": MulticlassF1Score(top_k=1, num_classes=int(num_classes), average="micro") for k in ks
        }
    )
    metrics.update(
        {
            f"top-{k}_f1_macro": MulticlassF1Score(top_k=1, num_classes=int(num_classes), average="macro") for k in ks
        }
    )
    metrics.update(
        {
            f"top-{k}_f1_weighted": MulticlassF1Score(top_k=1, num_classes=int(num_classes), average="weighted") for k in ks
        }
    )
    metrics.update(
        {
            f"top-{k}_f1_none": MulticlassF1Score(top_k=1, num_classes=int(num_classes), average="none") for k in ks
        }
    )
    # 计算auroc
    metrics.update(
        {
            f"top-{k}_auroc_macro": MulticlassAUROC(num_classes=int(num_classes), average="macro") for k in ks
        }
    )
    metrics.update(
        {
            f"top-{k}_auroc_weighted": MulticlassAUROC(num_classes=int(num_classes), average="weighted") for k in ks
        }
    )
    metrics.update(
        {
            f"top-{k}_auroc_none": MulticlassAUROC(num_classes=int(num_classes), average="none") for k in ks
        }
    )
    # 计算roc曲线
    metrics.update(
        {
            f"top-{k}_roc_macro": MulticlassROC(num_classes=int(num_classes), average="macro") for k in ks
        }
    )
    return MetricCollection(metrics)




def build_topk_imagenet_real_accuracy_metric(num_classes: int, ks: tuple = (1, 5)):
    metrics: Dict[str, Metric] = {f"top-{k}": ImageNetReaLAccuracy(top_k=k, num_classes=int(num_classes)) for k in ks}
    return MetricCollection(metrics)


class ImageNetReaLAccuracy(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.top_k = top_k
        self.add_state("tp", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        # preds [B, D]
        # target [B, A]
        # preds_oh [B, D] with 0 and 1
        # select top K highest probabilities, use one hot representation
        preds_oh = select_topk(preds, self.top_k)
        # target_oh [B, D + 1] with 0 and 1
        target_oh = torch.zeros((preds_oh.shape[0], preds_oh.shape[1] + 1), device=target.device, dtype=torch.int32)
        target = target.long()
        # for undefined targets (-1) use a fake value `num_classes`
        target[target == -1] = self.num_classes
        # fill targets, use one hot representation
        target_oh.scatter_(1, target, 1)
        # target_oh [B, D] (remove the fake target at index `num_classes`)
        target_oh = target_oh[:, :-1]
        # tp [B] with 0 and 1
        tp = (preds_oh * target_oh == 1).sum(dim=1)
        # at least one match between prediction and target
        tp.clip_(max=1)
        # ignore instances where no targets are defined
        mask = target_oh.sum(dim=1) > 0
        tp = tp[mask]
        self.tp.append(tp)  # type: ignore

    def compute(self) -> Tensor:
        tp = dim_zero_cat(self.tp)  # type: ignore
        return tp.float().mean()
