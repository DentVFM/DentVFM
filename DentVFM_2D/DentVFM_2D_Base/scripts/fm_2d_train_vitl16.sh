# 使用9h_fm_data_2d_v1测试训练
# PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=8 dinov2/train/train.py --config-file=dinov2/configs/train/vitl16_fm_2d_test.yaml --output-dir=../../../Model_Outputs/dinov2/Fm_2d_Test0001

# 使用9h_fm_data_2d_v2测试训练
# PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=8 dinov2/train/train.py --config-file=dinov2/configs/train/vitl16_fm_2d_test.yaml --output-dir=../../../Model_Outputs/dinov2/Fm_2d_Test0002

# 使用9h_fm_data_2d_v3测试训练
# PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=8 dinov2/train/train.py --config-file=dinov2/configs/train/vitl16_fm_2d_test.yaml --output-dir=../../../Model_Outputs/dinov2/Fm_2d_Test0004

# 使用9h_fm_data_2d_v3测试训练
# PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=8 dinov2/train/train.py --config-file=dinov2/configs/train/vitl16_fm_2d_test.yaml --output-dir=../../../Model_Outputs/dinov2/Fm_2d_Test0004

# 使用9h_fm_data_2d_v5测试训练
# PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=8 dinov2/train/train.py --config-file=dinov2/configs/train/vitl16_fm_2d_test.yaml --output-dir=../../../Model_Outputs/dinov2/Fm_2d_Test0005

# 使用9h_fm_data_2d_v6测试训练 vitl16
# PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=8 dinov2/train/train.py --config-file=dinov2/configs/train/vitl16_fm_2d_train.yaml --output-dir=../../../Model_Outputs/dinov2/Fm_2d_Test0007

# 使用9h_fm_data_2d_v8测试训练 vitl16
# PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=8 dinov2/train/train.py --config-file=dinov2/configs/train/vitl16_fm_2d_train.yaml --output-dir=../../../Model_Outputs/dinov2/Fm_2d_Test0007-2

# 使用9h_fm_data_2d_v9测试训练 vitl16
PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=8 dinov2/train/train.py --config-file=dinov2/configs/train/vitl16_fm_2d_train.yaml --output-dir=../../../Model_Outputs/dinov2/Fm_2d_Test0007-3

# 使用9h_fm_data_2d_v6测试训练
# PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=8 dinov2/train/train.py --config-file=dinov2/configs/train/vitg14_fm_2d_train.yaml --output-dir=../../../Model_Outputs/dinov2/Fm_2d_Test0008

# 使用9h_fm_data_2d_v6测试训练 vitl14
# PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=8 dinov2/train/train.py --config-file=dinov2/configs/train/vitl14_fm_2d_train.yaml --output-dir=../../../Model_Outputs/dinov2/Fm_2d_Test0010

# PYTHONPATH=. python dinov2/run/train/train.py \
#     --nodes 4 \
#     --config-file dinov2/configs/train/vitl16_fm_2d_test.yaml \
#     --output-dir ../../../Model_Outputs/dinov2/Fm_2d_Test0003 \
#     train.dataset_path=ImageNet:split=TRAIN:root=../../../Origin_Datasets/9h_fm_data_2d_v3/Data:extra=../../../Origin_Datasets/9h_fm_data_2d_v3/Extra