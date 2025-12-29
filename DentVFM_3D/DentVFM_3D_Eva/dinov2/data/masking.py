# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import random
import numpy as np


class MaskingGenerator3d:

    def __init__(
        self,
        input_size
    ):
        """
        Create a masking generator for 3D data, uses uniform random sampling to mask patches.
        Args:
            input_size: Size of the input data.
        """
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 3
        self.height, self.width, self.depth = input_size
        self.num_patches = self.height * self.width * self.depth

    def __repr__(self):
        repr_str = "Generator(%d, %d, %d)" % (
            self.height,
            self.width,
            self.depth
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width, self.depth

    def _mask(self, mask, n_masked):

        mask_inds = random.sample(range(self.num_patches), k=n_masked)
        mask.ravel()[mask_inds] = 1

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        self._mask(mask, num_masking_patches)
        return mask
