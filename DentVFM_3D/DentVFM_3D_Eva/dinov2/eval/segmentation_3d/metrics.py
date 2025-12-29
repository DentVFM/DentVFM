# Author: Tony Xu
#
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, Activations
from monai.data import decollate_batch


class BTCVMetrics:

    def __init__(self):
        self.post_label = AsDiscrete(to_onehot=14)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=14)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)

    def __call__(self, pred, target):
        target_list = decollate_batch(target)
        target_list = [self.post_label(t) for t in target_list]
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.dice_metric(y_pred=pred_list, y=target_list)
        self.dice_metric_batch(y_pred=pred_list, y=target_list)

        avg_dice = self.dice_metric.aggregate().item()
        class_dice = self.dice_metric_batch.aggregate()
        class_dice = [d.item() for d in class_dice]

        self.dice_metric.reset()
        self.dice_metric_batch.reset()

        return avg_dice, class_dice


class BraTSMetrics:

    def __init__(self):
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    def __call__(self, pred, target):
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.dice_metric(y_pred=pred_list, y=target)
        self.dice_metric_batch(y_pred=pred_list, y=target)

        avg_dice = self.dice_metric.aggregate().item()
        class_dice = self.dice_metric_batch.aggregate()
        metric_tc = class_dice[0].item()
        metric_wt = class_dice[1].item()
        metric_et = class_dice[2].item()

        self.dice_metric.reset()
        self.dice_metric_batch.reset()

        return avg_dice, (metric_tc, metric_wt, metric_et)


class LASEGMetrics(BTCVMetrics):

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=2)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)
        
        
# 此处修改，防止预测结果包含nan
class TF2Metrics():

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=49)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=49)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
        self.dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True)
    
    def __call__(self, pred, target):
        target_list = decollate_batch(target)
        target_list = [self.post_label(t) for t in target_list]
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.dice_metric(y_pred=pred_list, y=target_list)
        self.dice_metric_batch(y_pred=pred_list, y=target_list)

        avg_dice = self.dice_metric.aggregate()[0].item()
        class_dice, nan_flag = self.dice_metric_batch.aggregate()
        out_cls_dice = [0.0]
        for idx,d in enumerate(class_dice):
            if nan_flag[idx]>0:
                out_cls_dice.append(d.item())
            else:
                out_cls_dice.append(-1.0)

        self.dice_metric.reset()
        self.dice_metric_batch.reset()

        return avg_dice, out_cls_dice

    
class HNMetrics(BTCVMetrics):

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=10)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=10)

class PulMetrics(BTCVMetrics):

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=18)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=18)

# 此处修改
class TF3Metrics(TF2Metrics):

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=75)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=75)


# 此处修改，添加新的评价
def get_metric(dataset_name):
    if dataset_name == "BTCV":
        return BTCVMetrics()
    elif dataset_name == "BraTS":
        return BraTSMetrics()
    elif dataset_name == "LA-SEG":
        return LASEGMetrics()
    elif dataset_name == "TDSC-ABUS":
        return LASEGMetrics()  # same as LA-SEG
    
    # 此处修改，添加新的评测指标
    elif 'Dataset112_ToothFairy2' in dataset_name:
        return TF2Metrics()
    
    elif 'Dataset113_NCseg' in dataset_name:
        return LASEGMetrics()
    
    elif 'Dataset114_Sts_miccia23' in dataset_name:
        return TF2Metrics()
    
    elif 'Dataset115_Pal' in dataset_name:
        return LASEGMetrics()

    elif 'Dataset116_HeadNeck_miccai15' in dataset_name:
        return HNMetrics()

    elif 'Dataset117_Pulpy3D' in dataset_name:
        return PulMetrics()

    elif 'Dataset118_ToothFairy3' in dataset_name:
        return TF3Metrics()
    
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
