# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

from torch.utils.data import Dataset


class DictDatasetWithEnumeratedTargets(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, index: int) -> Tuple[Any, Tuple[Any, int]]:
        data_dict = self._dataset[index]
        image = data_dict["image"]
        target = data_dict.get("label", None)
        target = index if target is None else target
        return image, (index, target)

    def __len__(self) -> int:
        return len(self._dataset)
