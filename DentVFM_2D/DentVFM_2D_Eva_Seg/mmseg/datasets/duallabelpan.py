# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import random

@DATASETS.register_module()
class duallabelpan(BaseSegDataset):
    """STARE dataset.

    In segmentation map annotation for STARE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.ah.png'.
    """
    METAINFO = dict(
        classes=('background', 
                 '11', '12', '13', '14', '15', '16', '17', '18',
                 '21', '22', '23', '24', '25', '26', '27', '28',
                 '31', '32', '33', '34', '35', '36', '37', '38',
                 '41', '42', '43', '44', '45', '46', '47', '48'),
        palette=[[0,0,0], [236, 224, 14], [116, 117, 42], [212, 168, 186], 
                 [185, 81, 227], [20, 39, 121], [243, 126, 218], [164, 53, 144], 
                 [46, 234, 59], [68, 223, 1], [63, 7, 69], [23, 41, 101], [172, 67, 125], 
                 [8, 249, 169], [101, 221, 72], [36, 140, 162], [134, 28, 99], [239, 50, 48], 
                 [184, 117, 56], [141, 146, 15], [242, 232, 36], [136, 225, 66], [209, 168, 231], 
                 [7, 106, 38], [237, 131, 143], [82, 61, 144], [113, 158, 253], [41, 221, 231], 
                 [232, 177, 65], [36, 85, 205], [138, 47, 172], [56, 5, 206], [109, 94, 69]])

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
