# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class dentalbitewing(BaseSegDataset):
    """STARE dataset.

    In segmentation map annotation for STARE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.ah.png'.
    """
    METAINFO = dict(
        classes=('background','Missing','bone','cavity','crown','dental-implant','dental-implant-crown',
                 'dentin','enamel','filling-metal','filling-non-metal','periapical-radiolucence','pulp','root canal',
                 'sinus'),
        palette=[[0,0,0],
                [131,224,112],
                [215,179,255],
                [246,51,81],
                [58,132,255],
                [134,202,218],
                [221,195,130],
                [255,255,127],
                [255,255,255],
                [1,13,27],
                [0,133,255],
                [24,250,143],
                [255,105,248],
                [17,253,231],
                [255,146,119]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_mask.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        assert fileio.exists(
            self.data_prefix['img_path'], backend_args=self.backend_args)
