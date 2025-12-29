# 评测下游任务：全景片分类是否需要手术 apicoectomy_surgery_panoramic
# 该脚本适用服务器集群多卡并行评测

# 数据集参数
dataset_name="apicoectomy_surgery_panoramic_cls_v2"
train_val_test_nums="68 68 68"
# base_dir="/mnt/ugreenf8c5_samba_share/huangxinrui"
base_dir="/home/jovyan/dataset"

# 模型参数
# model_dirs=("ILSVRC_2012_Test0001" "Dinov2_Official_Model_vitl14" "Dinov2_Official_Model_vitg14" "Fm_2d_Test0001" "Fm_2d_Test0002" "Fm_2d_Test0003" "Fm_2d_Test0004" "Fm_2d_Test0005" "Fm_2d_Test0006" "Fm_2d_Test0006-1" "Fm_2d_Test0006-2" "Fm_2d_Test0006-3" "Fm_2d_Test0006-4" "Fm_2d_Test0006-5" "Fm_2d_Test0006-6" "Fm_2d_Test0006-7" "Fm_2d_Test0006-8" "Fm_2d_Test0006-9" "Fm_2d_Test0007" "Fm_2d_Test0007-1" "Fm_2d_Test0008" "Fm_2d_Test0008-1" "Fm_2d_Test0008-2" "Fm_2d_Test0008-3" "Fm_2d_Test0010" "Fm_2d_Test0010-1" "Fm_2d_Test0010-2" "Fm_2d_Test0010-3")
model_dirs=("Fm_2d_Test0008-3")

# 全局参数
max_jobs=40

# KNN评测
batch_size="128"
for model_dir in "${model_dirs[@]}"
do 
    for ckp_dir in $base_dir/Model_Outputs/dinov2/$model_dir/eval/*/
    do
        # 统计当前有几个正在运行的jobs
        running_count=$(baizectl job ls -t PYTORCH | awk '$1 ~ /^task/ && $3 == "RUNNING"' | wc -l)

        # 如果当前正在运行的job数量大于限定值，则会等待，否则启动新的job
        while [ "$running_count" -ge "$max_jobs" ]; do
            echo "waiting..."
            sleep 10
            running_count=$(baizectl job ls -t PYTORCH | awk '$1 ~ /^task/ && $3 == "RUNNING"' | wc -l)
        done

        # 提交当前job
        ckp_name=$(basename "$ckp_dir")
        echo "[KNN Evaluation] Now is processing model_dir: $model_dir/$ckp_name"
        current_time=$(date +"%Y%m%d%H%M%S")
        job_name="task-knn-${model_dir}"
        job_name="${job_name//_/-}"
        job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
        job_name="${job_name}-${current_time}"
        baizectl job submit --shm-size 1024 \
                    --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan15 \
                    --name $job_name \
                    --max-retries 1 \
                    --workers 1 \
                    --requests-resources nvidia.com/gpucores=20,nvidia.com/gpumem=20k,nvidia.com/vgpu=1 \
                    --resources nvidia.com/gpucores=20,nvidia.com/gpumem=20k,nvidia.com/vgpu=1 \
                    -- bash -c "PYTHONPATH=. python dinov2/eval/knn.py \
                                --config-file $base_dir/Model_Outputs/dinov2/$model_dir/config.yaml \
                                --pretrained-weights $base_dir/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/teacher_checkpoint.pth \
                                --output-dir $base_dir/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/knn_$dataset_name \
                                --train-dataset ImageNet:split=TRAIN:root=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/apicoectomy_surgery_panoramic/$dataset_name:extra=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/apicoectomy_surgery_panoramic/$dataset_name/Extra \
                                --val-dataset ImageNet:split=VAL:root=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/apicoectomy_surgery_panoramic/$dataset_name:extra=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/apicoectomy_surgery_panoramic/$dataset_name/Extra \
                                --nb_knn 10 15 20 \
                                --batch_size $batch_size \
                                --save_exist \
                                --train_val_test_nums $train_val_test_nums > ./task_evaluation_logs/${dataset_name}/knn_${model_dir}_${ckp_name}.txt 2>&1"
        # 此处等待程序启动
        sleep 3
    done
done


# Logistic Regression评测
batch_size="128"
for model_dir in "${model_dirs[@]}"
do 
    for ckp_dir in $base_dir/Model_Outputs/dinov2/$model_dir/eval/*/
    do
        # 统计当前有几个正在运行的jobs
        running_count=$(baizectl job ls -t PYTORCH | awk '$1 ~ /^task/ && $3 == "RUNNING"' | wc -l)

        # 如果当前正在运行的job数量大于限定值，则会等待，否则启动新的job
        while [ "$running_count" -ge "$max_jobs" ]; do
            echo "waiting..."
            sleep 10
            running_count=$(baizectl job ls -t PYTORCH | awk '$1 ~ /^task/ && $3 == "RUNNING"' | wc -l)
        done

        # 提交当前job
        ckp_name=$(basename "$ckp_dir")
        echo "[LogisticR Evaluation] Now is processing model_dir: $model_dir/$ckp_name"
        current_time=$(date +"%Y%m%d%H%M%S")
        job_name="task-logr-${model_dir}"
        job_name="${job_name//_/-}"
        job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
        job_name="${job_name}-${current_time}"
        baizectl job submit --shm-size 1024 \
                    --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan15 \
                    --name $job_name \
                    --max-retries 1 \
                    --workers 1 \
                    --requests-resources nvidia.com/gpucores=20,nvidia.com/gpumem=20k,nvidia.com/vgpu=1 \
                    --resources nvidia.com/gpucores=20,nvidia.com/gpumem=20k,nvidia.com/vgpu=1 \
                    -- bash -c "PYTHONPATH=. python dinov2/eval/log_regression.py \
                                --config-file $base_dir/Model_Outputs/dinov2/$model_dir/config.yaml \
                                --pretrained-weights $base_dir/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/teacher_checkpoint.pth \
                                --output-dir $base_dir/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/logreg_$dataset_name \
                                --train-dataset ImageNet:split=TRAIN:root=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/apicoectomy_surgery_panoramic/$dataset_name:extra=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/apicoectomy_surgery_panoramic/$dataset_name/Extra \
                                --val-dataset ImageNet:split=VAL:root=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/apicoectomy_surgery_panoramic/$dataset_name:extra=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/apicoectomy_surgery_panoramic/$dataset_name/Extra \
                                --batch_size $batch_size \
                                --save_exist \
                                --train_val_test_nums $train_val_test_nums > ./task_evaluation_logs/${dataset_name}/logr_${model_dir}_${ckp_name}.txt 2>&1"
        
        # 此处等待程序启动
        sleep 3
    done
done




# 等待全部任务完成
running_count=$(baizectl job ls -t PYTORCH | awk '$1 ~ /^task/ && $3 == "RUNNING"' | wc -l)
while [ "$running_count" -gt 0 ]; do
            echo "waiting..."
            sleep 10
            running_count=$(baizectl job ls -t PYTORCH | awk '$1 ~ /^task/ && $3 == "RUNNING"' | wc -l)
        done

echo "All Programs Have Completed !"