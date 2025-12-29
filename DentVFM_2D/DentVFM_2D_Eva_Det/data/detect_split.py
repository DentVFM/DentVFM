import json
from pathlib import Path
from typing import List

import mmcv
import numpy as np
from sklearn.model_selection import ShuffleSplit
from skmultilearn.model_selection import iterative_train_test_split

from dataset import RadiographDataset


def multi2onehot(multi_label):
    out = np.zeros(len(RadiographDataset.CLASSES))
    out[multi_label] = 1
    
    return out


def split_patients(
    root: Path,
    exclude_dirs: List[str]=['False Neg', 'False Pos'],
):
    # determine multi-label of each image
    files = list(root.glob('annotations/*.txt'))
    files = sorted(files)


    for exclude_dir in exclude_dirs:
        exclude_files = list(root.glob(f'**/{exclude_dir}/*'))
        exclude_ids = [f.stem for f in exclude_files]
        files = [f for f in files if f.stem not in exclude_ids]


    ys = np.empty((len(files), len(RadiographDataset.CLASSES)))
    for i, file in enumerate(files):
        anns = mmcv.list_from_file(file)
        bbox_classes = [ann.split(' ')[0] for ann in anns]
        bbox_classes = [c for c in bbox_classes if c in RadiographDataset.CLASSES]
        bbox_labels = [RadiographDataset.CLASSES.index(c) for c in bbox_classes]
        ys[i] = multi2onehot(bbox_labels)
        
    # split in train and validation-test splits
    image_ids = np.array([f.name[:-4] for f in files])
    idxs = np.arange(len(files)).reshape(-1, 1)
    split = iterative_train_test_split(idxs, ys, test_size=0.2)
    X_train, y_train, X_val_test, y_val_test = split

    with open(root / 'train.txt', 'w') as f:
        for image_id in image_ids[X_train[:, 0]]:
            f.write(image_id + '\n')

    # further split in validation and test splits
    split = iterative_train_test_split(X_val_test, y_val_test, test_size=0.5)
    X_val, y_val, X_test, y_test = split

    with open(root / 'val.txt', 'w') as f:
        for image_id in image_ids[X_val[:, 0]]:
            f.write(image_id + '\n')

    with open(root / 'test.txt', 'w') as f:
        for image_id in image_ids[X_test[:, 0]]:
            f.write(image_id + '\n')

    # determine number of individuals in each split
    return X_train.shape[0], X_val.shape[0], X_test.shape[0]


def control_images(root: Path):
    image_ids = []
    for ann_file in root.glob('annotations/pan*.json'):
        with open(ann_file, 'rb') as f:
            json_dict = json.load(f)

        for image_id, image_dict in json_dict.items():
            if 'healthy' in image_dict['diagnoses']:
                image_ids.append(image_id)
    
    return sorted(image_ids)


def split_controls(root: Path, image_ids, num_train, num_val, num_test):
    rs = ShuffleSplit(
        n_splits=1,
        train_size=num_train,
        test_size=num_val + num_test,
        random_state=1234,
    )
    train_idxs, val_test_idxs = next(rs.split(image_ids))

    with open(root / 'train.txt', 'a') as f:
        for idx in train_idxs:
            f.write(image_ids[idx] + '\n')

    rs = ShuffleSplit(
        n_splits=1,
        train_size=num_val,
        test_size=num_test,
        random_state=1234,
    )
    val_idxs, test_idxs = next(rs.split(val_test_idxs))

    with open(root / 'val.txt', 'a') as f:
        for idx in val_idxs:
            f.write(image_ids[idx] + '\n')

    with open(root / 'test.txt', 'a') as f:
        for idx in test_idxs:
            f.write(image_ids[idx] + '\n')


if __name__ == '__main__':
    np.random.seed(1234)

    root = Path('/mnt/diag/opgs/Osteolysis_Carotis')
    num_train, num_val, num_test = split_patients(root, ['False Pos'])

    use_controls = False
    if use_controls:
        image_ids = control_images(root)
        split_controls(root, image_ids, num_train, num_val, num_test)
