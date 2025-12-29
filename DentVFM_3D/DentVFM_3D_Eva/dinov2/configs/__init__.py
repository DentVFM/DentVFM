# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import pathlib

from omegaconf import OmegaConf


def load_config(config_name: str):
    config_filename = config_name + ".yaml"
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / config_filename)


dinov2_default_config_3d = load_config("ssl3d_default_config")


def load_and_merge_config_3d(config_name: str):
    default_config = OmegaConf.create(dinov2_default_config_3d)
    loaded_config = load_config(config_name)
    return OmegaConf.merge(default_config, loaded_config)

