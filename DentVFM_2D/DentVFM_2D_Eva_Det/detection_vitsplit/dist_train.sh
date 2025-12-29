#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28505}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    ./train_new.py $CONFIG --launcher pytorch ${@:3}
