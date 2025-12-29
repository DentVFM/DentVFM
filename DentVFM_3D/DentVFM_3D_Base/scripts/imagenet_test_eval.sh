# 该脚本用于模型评价
# 此处预训练模型使用imagenet-1k训练


# evaluate imagenet-1k
# PYTHONPATH=. python dinov2/eval/knn.py \
#     --config-file ../../../Model_Outputs/dinov2/ILSVRC_2012_Test0001/config.yaml \
#     --pretrained-weights ../../../Model_Outputs/dinov2/ILSVRC_2012_Test0001/eval/training_124999/teacher_checkpoint.pth \
#     --output-dir ../../../Model_Outputs/dinov2/ILSVRC_2012_Test0001/eval/training_124999/knn \
#     --train-dataset ImageNet:split=TRAIN:root=../../../Origin_Datasets/ILSVRC_2012/Data/CLS-LOC:extra=../../../Origin_Datasets/ILSVRC_2012/extra \
#     --val-dataset ImageNet:split=VAL:root=../../../Origin_Datasets/ILSVRC_2012/Data/CLS-LOC:extra=../../../Origin_Datasets/ILSVRC_2012/extra \
#     --batch-size 512
    
# evaluate modality_recognition_2d_xray_only
PYTHONPATH=. python dinov2/eval/knn.py \
    --config-file ../../../Model_Outputs/dinov2/ILSVRC_2012_Test0001/config.yaml \
    --pretrained-weights ../../../Model_Outputs/dinov2/ILSVRC_2012_Test0001/eval/training_49999/teacher_checkpoint.pth \
    --output-dir ../../../Model_Outputs/dinov2/ILSVRC_2012_Test0001/eval/training_49999/knn_modality_recognition_2d_xray_only \
    --train-dataset ImageNet:split=TRAIN:root=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/modality_recognition/modality_recognition_2d_xray_only_v5:extra=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/modality_recognition/modality_recognition_2d_xray_only_v5/extra \
    --val-dataset ImageNet:split=VAL:root=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/modality_recognition/modality_recognition_2d_xray_only_v5:extra=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/modality_recognition/modality_recognition_2d_xray_only_v5/extra \
    --batch-size 32