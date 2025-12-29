# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import submitit

from dinov2.utils.cluster import (
    get_slurm_executor_parameters,
    get_slurm_partition,
    get_user_checkpoint_path,
)


logger = logging.getLogger("dinov2")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
) -> argparse.ArgumentParser:
    parents = parents or []
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--ngpus",
        "--gpus",
        "--gpus-per-node",
        default=8,
        type=int,
        help="Number of GPUs to request on each node",
    )
    parser.add_argument(
        "--nodes",
        "--nnodes",
        default=1,
        type=int,
        help="Number of nodes to request",
    )
    return parser


def submit_jobs(args):
    if not args.output_dir:
        raise RuntimeError("Path to user checkpoint cannot be determined")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Submitted job")
    str_output_dir = os.path.abspath(args.output_dir)
    logger.info(f"Logs and checkpoints will be saved at: {str_output_dir}")
