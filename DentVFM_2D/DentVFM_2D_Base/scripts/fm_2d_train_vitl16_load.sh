# 此处需要加载预训练权重初始化

# 使用9h_fm_data_2d_v6, v9测试训练
PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=8 dinov2/train/train.py --config-file=dinov2/configs/train/vitl16_fm_2d_load_train.yaml --output-dir=../../../Model_Outputs/dinov2/Fm_2d_Test0007-4