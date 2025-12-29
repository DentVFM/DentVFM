import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit

from dataset import RadiographDataset


def split_patients(
    files: List[Path],
) -> Tuple[List[int], List[int], List[int]]:
    num_train, num_val, num_test = [], [], []

    kf = KFold(n_splits=10, shuffle=True, random_state=1234)
    for i, (train_val_idxs, test_idxs) in enumerate(kf.split(files)):
        train_val_idxs = np.random.permutation(train_val_idxs)
        idx = int(train_val_idxs.shape[0] * 0.89)
        train_idxs = train_val_idxs[:idx]
        val_idxs = train_val_idxs[idx:]

        image_ids = np.array([f.stem for f in files])
        with open(root / f'traini{i}.txt', 'w') as f:
            for image_id in image_ids[train_idxs]:
                f.write(image_id + '\n')

        with open(root / f'vali{i}.txt', 'w') as f:
            for image_id in image_ids[val_idxs]:
                f.write(image_id + '\n')

        with open(root / f'testi{i}.txt', 'w') as f:
            for image_id in image_ids[test_idxs]:
                f.write(image_id + '\n')

        num_train.append(train_idxs.shape[0])
        num_val.append(val_idxs.shape[0])
        num_test.append(test_idxs.shape[0])

    # determine number of individuals in each split
    return num_train, num_val, num_test


def control_images(root: Path):
    image_ids = []
    for ann_file in root.glob('annotations/pan*.json'):
        with open(ann_file, 'rb') as f:
            json_dict = json.load(f)

        for image_id, image_dict in json_dict.items():
            if 'healthy' in image_dict['diagnoses']:
                image_ids.append(image_id)
    
    return sorted(image_ids)


def final_control_images(root: Path):
    image_ids_set = set()
    for fold in range(0, 10):
        with open(root / f'test{fold}.txt', 'r') as f:
            image_ids = [l.strip() for l in f.readlines()]
            image_ids_set = image_ids_set | set(image_ids)
        with open(root / f'val{fold}.txt', 'r') as f:
            image_ids = [l.strip() for l in f.readlines()]
            image_ids_set = image_ids_set | set(image_ids)
        with open(root / f'train{fold}.txt', 'r') as f:
            image_ids = [l.strip() for l in f.readlines()]
            image_ids_set = image_ids_set | set(image_ids)

    image_ids = []
    for image_id in image_ids_set:
        if (root / 'annotations' / f'{image_id}.txt').exists():
            continue

        image_ids.append(image_id)

    return sorted(image_ids)


def split_controls(root: Path, image_ids, num_train, num_val, num_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=1234)

    for i, (train_val_idxs, test_idxs) in enumerate(kf.split(image_ids)):
        train_val_idxs = np.random.permutation(train_val_idxs)
        idx = int(train_val_idxs.shape[0] * 0.89)
        train_idxs = train_val_idxs[:idx]
        val_idxs = train_val_idxs[idx:]

        with open(root / f'traini{i}.txt', 'a') as f:
            for idx in train_idxs[:num_train[i]]:
                f.write(image_ids[idx] + '\n')

        with open(root / f'vali{i}.txt', 'a') as f:
            for idx in val_idxs[:num_val[i]]:
                f.write(image_ids[idx] + '\n')

        with open(root / f'testi{i}.txt', 'a') as f:
            for idx in test_idxs[:num_test[i]]:
                f.write(image_ids[idx] + '\n')


if __name__ == '__main__':
    np.random.seed(1234)

    root = Path('/mnt/diag/opgs/Osteolysis_Carotis')

    # determine multi-label of each image
    files = list(root.glob('annotations/*.txt'))
    files = sorted(files)

    for exclude_dir in ['False Pos', 'False Neg']:
        exclude_files = list(root.glob(f'**/{exclude_dir}/*'))
        exclude_ids = [f.stem for f in exclude_files]
        files = [f for f in files if f.stem not in exclude_ids]

    num_train, num_val, num_test = split_patients(files)

    use_controls = True
    if use_controls:
        image_ids = final_control_images(root)
        split_controls(root, image_ids, num_train, num_val, num_test)
