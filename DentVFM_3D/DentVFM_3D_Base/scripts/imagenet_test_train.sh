# 使用ImageNet-1K测试训练
# PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=8 dinov2/train/train.py --config-file=dinov2/configs/train/vitl16_imagenet_test.yaml --output-dir=../../../Model_Outputs/dinov2/ILSVRC_2012_Test0001

# 测试断点重训---同上命令
PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=8 dinov2/train/train.py --config-file=dinov2/configs/train/vitl16_imagenet_test.yaml --output-dir=../../../Model_Outputs/dinov2/ILSVRC_2012_Test0001