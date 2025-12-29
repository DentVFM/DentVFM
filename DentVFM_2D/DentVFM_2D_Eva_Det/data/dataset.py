import os.path as osp

import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
import numpy as np


@DATASETS.register_module()
class RadiographDataset(CustomDataset):

    CLASSES = (
        # 'calcified_stylohyoid_ligament',
        # 'cornu_superior',
        'carotis_plaques',
        # 'tonsilolith',
        # 'phlebolith',
        # 'cartilago_triticea',
        # 'sialolites',
        # 'calcified_lymph_node',
    )

    def load_annotations(self, ann_file):
        class2label = {k: i for i, k in enumerate(self.CLASSES)}
    
        # convert annotations to middle format
        annotations = []
        for image_id in mmcv.list_from_file(ann_file):
            # print(image_id)
            # load image
            file = osp.join(self.img_prefix, f'{image_id}.jpg')
            img = mmcv.imread(file)
            height, width = img.shape[:2]
    
            # load annotations
            label_prefix = self.img_prefix.replace('radiographs', 'annotations')
            file = osp.join(label_prefix, f'{image_id}.txt')
            if osp.exists(file):
                lines = mmcv.list_from_file(file)
    
                content = [line.strip().split(' ') for line in lines]
                bbox_classes = [x[0] for x in content]
                bbox_coordinates = [[float(c) for c in x[1:5]] for x in content]
            else:
                bbox_classes, bbox_coordinates = [], []
    
            # filter bboxes with class not in self.CLASSES as 'background'
            labels, labels_ignore = np.empty((2, 0))
            bboxes, bboxes_ignore = np.empty((2, 0, 4))
            for bbox_class, bbox in zip(bbox_classes, bbox_coordinates):
                if bbox_class in self.CLASSES:
                    bbox_label = class2label[bbox_class]
                    labels = np.concatenate((labels, [bbox_label]))
                    bboxes = np.concatenate((bboxes, [bbox]))
                else:
                    labels_ignore = np.concatenate((labels_ignore, [-1]))
                    bboxes_ignore = np.concatenate((bboxes_ignore, [bbox]))
    
            # convert  middle format and add to list of annotations
            annotations.append({
                'filename': f'{image_id}.jpg',
                'width': width,
                'height': height,
                'ann': {
                    'labels': labels.astype(np.int64),
                    'bboxes': bboxes.astype(np.float32),
                    'labels_ignore': labels_ignore.astype(np.int64),
                    'bboxes_ignore': bboxes_ignore.astype(np.float32),
                }
            })

        return annotations