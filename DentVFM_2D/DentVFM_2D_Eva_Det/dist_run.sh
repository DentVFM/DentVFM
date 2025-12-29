#!/usr/bin/env bash

GPUS=$1
PORT=$2

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    train.py --launcher pytorch ${@:3}
