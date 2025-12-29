current_time=$(date +"%Y%m%d%H%M%S")
job_name="task-finetune-bmd-${current_time}"
# dataset_name="9h_BMD_grade_crop_dilation_20_v2_fold1"
dataset_name="9h_BMD_grade_crop_dilation_20_v1_fold1"
# train_val_test_nums="1044 269 269"
train_val_test_nums="618 269 269"
baizectl job submit --shm-size 10240 \
            --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan15 \
            --name $job_name \
            --restart-policy never \
            --workers 1 \
            --requests-resources nvidia.com/gpucores=100,nvidia.com/gpumem=81k,nvidia.com/vgpu=1 \
            --resources nvidia.com/gpucores=100,nvidia.com/gpumem=81k,nvidia.com/vgpu=1 \
            -- bash -c "HF_HOME='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint' HF_HUB_OFFLINE='1'\
                        PYTHONPATH=. python dinov2/eval/finetune_new.py \
                        --config-file /home/jovyan/dataset/Model_Outputs/dinov2/Resnet50/config.yaml \
                        --pretrained-weights /home/jovyan/dataset/Model_Outputs/dinov2/Resnet50/eval/training_0001/teacher_checkpoint.pth \
                        --output-dir /home/jovyan/dataset/Model_Outputs/dinov2/Resnet50/eval/training_0001/finetune_$dataset_name \
                        --train-dataset ImageNet:split=TRAIN:root=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/9h_plant_BMD_grade_classification/$dataset_name:extra=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/9h_plant_BMD_grade_classification/$dataset_name/Extra \
                        --val-dataset ImageNet:split=VAL:root=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/9h_plant_BMD_grade_classification/$dataset_name:extra=/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/9h_plant_BMD_grade_classification/$dataset_name/Extra \
                        --batch_size 128 \
                        --train_val_test_nums $train_val_test_nums > ./task_evaluation_logs/$dataset_name/finetune_resnet50_training_0001.txt 2>&1"