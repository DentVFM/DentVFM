# 该脚本用于模型评价
# 此处预训练模型使用9h_fm_data_2d_vx训练

    
# evaluate modality_recognition_2d_xray_only
for log_name in ILSVRC_2012_Test0001; do
    echo "Now is evaluating $log_name !"
    
    for iter_name in "../../../Model_Outputs/dinov2/$log_name/eval"/training_*; do
        echo "Now is evaluating $iter_name !"
        
        PYTHONPATH=. python dinov2/eval/knn.py \
        --config-file "../../../Model_Outputs/dinov2/$log_name/config.yaml" \
        --pretrained-weights "$iter_name/teacher_checkpoint.pth" \
        --output-dir "$iter_name/knn_modality_recognition_2d_xray_only" \
        --train-dataset "ImageNet:split=TRAIN:root=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/modality_recognition/modality_recognition_2d_xray_only_v5:extra=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/modality_recognition/modality_recognition_2d_xray_only_v5/extra" \
        --val-dataset "ImageNet:split=VAL:root=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/modality_recognition/modality_recognition_2d_xray_only_v5:extra=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/modality_recognition/modality_recognition_2d_xray_only_v5/extra" \
        --batch-size 32
        
    done
done

# PYTHONPATH=. python dinov2/eval/knn.py \
#         --config-file ../../../Model_Outputs/dinov2/Fm_2d_Test0002/config.yaml \
#         --pretrained-weights ../../../Model_Outputs/dinov2/Fm_2d_Test0002/eval/training_74999/teacher_checkpoint.pth \
#         --output-dir ../../../Model_Outputs/dinov2/Fm_2d_Test0002/eval/training_74999/knn_modality_recognition_2d_xray_only \
#         --train-dataset ImageNet:split=TRAIN:root=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/modality_recognition/modality_recognition_2d_xray_only_v5:extra=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/modality_recognition/modality_recognition_2d_xray_only_v5/extra \
#         --val-dataset ImageNet:split=VAL:root=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/modality_recognition/modality_recognition_2d_xray_only_v5:extra=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/modality_recognition/modality_recognition_2d_xray_only_v5/extra \
        # --batch-size 32