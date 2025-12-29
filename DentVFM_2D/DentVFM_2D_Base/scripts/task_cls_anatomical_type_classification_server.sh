# 评测下游任务：侧位片分类畸形类型 anatomical_type_classification
# 该脚本适用服务器集群多卡并行评测


# 数据集参数
kfold=5
# base_dir="/mnt/ugreenf8c5_samba_share/huangxinrui"
base_dir="/home/jovyan/dataset"

# 扩展包含foldx的数据集名
base_dataset_name="anatomical_type_classification_ANB_v2"
# dataset_name="anatomical_type_classification_ANB_v1"
dataset_names=()
for i in $(seq 1 $kfold)
do
    dataset_names+=("${base_dataset_name}_fold$i")
done
train_val_test_nums="167 169 169"
# train_val_test_nums="167 169 199"
# train_val_test_nums="197 199 199"


# 可视化相关信息
echo "[当前正在评测数据集] ${base_dataset_name}, 包含fold数据集: ${dataset_names[@]}"
echo "[当前评测数据集的训练集、验证集、测试集的数量] ${train_val_test_nums}"


# 模型参数
## VFM对比方法
model_dirs=("Resnet50" "Clip" "SAM" "Dinov2_Official_Model_vitb14" "Dinov2_Official_Model_vitl14" "Dinov2_Official_Model_vitg14" "Biomedclip" "SAM_Med2d" "LVM_Resnet50" "LVM_vitb16")
## SSL对比方法
# model_dirs=("MAE_2d_9h_vitb14" "MAE_2d_9h_vitg14" "MAE_2d_9h_vitl14" "MAE_2d_9h_vitl16" "iBoT_2d_9h_vitb14" "iBoT_2d_9h_vitl14")
# model_dirs=("Resnet50" "Clip" "Biomedclip" "SAM" "MedSAM" "SAM_Med2d" "LVM_Resnet50" "LVM_vitb16" "MAE_2d_9h_vitb14" "MAE_2d_9h_vitg14" "MAE_2d_9h_vitl14" "MAE_2d_9h_vitl16" "iBoT_2d_9h_vitb14" "iBoT_2d_9h_vitl14" "ILSVRC_2012_Test0001" "Dinov2_Official_Model_vitl14" "Dinov2_Official_Model_vitg14" "Fm_2d_Test0001" "Fm_2d_Test0002" "Fm_2d_Test0003" "Fm_2d_Test0004" "Fm_2d_Test0005" "Fm_2d_Test0006" "Fm_2d_Test0006-1" "Fm_2d_Test0006-2" "Fm_2d_Test0006-3" "Fm_2d_Test0006-4" "Fm_2d_Test0006-5" "Fm_2d_Test0006-6" "Fm_2d_Test0006-7" "Fm_2d_Test0006-8" "Fm_2d_Test0006-9" "Fm_2d_Test0007" "Fm_2d_Test0007-1" "Fm_2d_Test0008" "Fm_2d_Test0008-1" "Fm_2d_Test0008-2" "Fm_2d_Test0008-3" "Fm_2d_Test0010" "Fm_2d_Test0010-1" "Fm_2d_Test0010-2" "Fm_2d_Test0010-3" "Fm_2d_Test0011-1")
# model_dirs=("Resnet50" "Clip" "SAM" "Biomedclip" "SAM_Med2d" "LVM_Resnet50" "LVM_vitb16" "Fm_2d_Test0011-1" "Fm_2d_Test0010-2" "Fm_2d_Test0008-1")
# ckp_dirs=("training_0001" "training_vitb16" "training_vitb" "training_vitb16" "training_0001" "training_0001" "training_0001" "training_87499" "training_524999" "training_99999")


# KNN评测
# for dataset_name in "${dataset_names[@]}"
# do
#     echo "[开始评测每一折数据] ${dataset_name}"
#     max_jobs=30
#     gpucores="30"
#     gpumem="30k"
#     batch_size="128"
    
#     # 每一折针对每个数据集测试
#     echo "[开始每个模型测试] ${dataset_name}"
#     for model_dir in "${model_dirs[@]}"
#     do
#         # 如果是sam类的方法需要调整参数
#         if [ "${model_dir}" == "SAM" -o "${model_dir}" == "MedSAM" -o "${model_dir}" == "SAM_Med2d" ]; then
#             batch_size="16"
#             gpucores="100"
#             gpumem="81k"
#         fi

#         for ckp_dir in $base_dir/Model_Outputs/dinov2/$model_dir/eval/*/
#         do
#             # 统计当前有几个正在运行的jobs
#             running_count=$(baizectl job ls -t PYTORCH | awk '$1 ~ /^task/ && $3 == "RUNNING"' | wc -l)

#             # 如果当前正在运行的job数量大于限定值，则会等待，否则启动新的job
#             while [ "$running_count" -ge "$max_jobs" ]; do
#                 echo "waiting..."
#                 sleep 10
#                 running_count=$(baizectl job ls -t PYTORCH | awk '$1 ~ /^task/ && $3 == "RUNNING"' | wc -l)
#             done

#             # 提交当前job
#             ckp_name=$(basename "$ckp_dir")
#             echo "[KNN Evaluation] Now is processing model_dir: $model_dir/$ckp_name"
            
            
#             # 重新跑全部，因为此时fold1不是原本的fold
#             # echo "执行评测！"
#             # current_time=$(date +"%Y%m%d%H%M%S")
#             # job_name="task-knn-ana"
#             # job_name="${job_name//_/-}"
#             # job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
#             # job_name="${job_name}-${current_time}"

#             # # 创建log保存目录
#             # mkdir -p ./task_evaluation_logs/${dataset_name}

#             # baizectl job submit --shm-size 10240 \
#             #             --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan15 \
#             #             --name $job_name \
#             #             --max-retries 1 \
#             #             --workers 1 \
#             #             --requests-resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
#             #             --resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
#             #             -- bash -c "HF_HOME='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint' HF_HUB_OFFLINE='1'\
#             #                         PYTHONPATH=. python dinov2/eval/knn_new.py \
#             #                         --config-file $base_dir/Model_Outputs/dinov2/$model_dir/config.yaml \
#             #                         --pretrained-weights $base_dir/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/teacher_checkpoint.pth \
#             #                         --output-dir $base_dir/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/knn_$dataset_name \
#             #                         --train-dataset ImageNet:split=TRAIN:root=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name:extra=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name/Extra \
#             #                         --val-dataset ImageNet:split=VAL:root=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name:extra=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name/Extra \
#             #                         --nb_knn 10 15 20 \
#             #                         --batch_size $batch_size \
#             #                         --train_val_test_nums $train_val_test_nums > ./task_evaluation_logs/${dataset_name}/knn_${model_dir}_${ckp_name}.txt 2>&1"
#             # # 此处等待程序启动
#             # sleep 3
            
            
            
            
#             # 判断当前是否已经存在该模型的评测结果
#             FILE="$base_dir/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/knn_$dataset_name/results_eval_knn.json"
#             if [ -f "$FILE" ]; then
#                 # 如果文件存在
#                 echo "发现当前文件存在！跳过重新评测！"
#             else
#                 # 如果文件不存在
#                 echo "发现当前文件不存在，执行评测！"
#                 current_time=$(date +"%Y%m%d%H%M%S")
#                 job_name="task-knn-ana"
#                 job_name="${job_name//_/-}"
#                 job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
#                 job_name="${job_name}-${current_time}"
                
#                 # 创建log保存目录
#                 mkdir -p ./task_evaluation_logs/${dataset_name}
                
#                 baizectl job submit --shm-size 10240 \
#                             --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan15 \
#                             --name $job_name \
#                             --max-retries 1 \
#                             --workers 1 \
#                             --requests-resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
#                             --resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
#                             -- bash -c "HF_HOME='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint' HF_HUB_OFFLINE='1'\
#                                         PYTHONPATH=. python dinov2/eval/knn_new.py \
#                                         --config-file $base_dir/Model_Outputs/dinov2/$model_dir/config.yaml \
#                                         --pretrained-weights $base_dir/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/teacher_checkpoint.pth \
#                                         --output-dir $base_dir/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/knn_$dataset_name \
#                                         --train-dataset ImageNet:split=TRAIN:root=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name:extra=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name/Extra \
#                                         --val-dataset ImageNet:split=VAL:root=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name:extra=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name/Extra \
#                                         --nb_knn 10 15 20 \
#                                         --batch_size $batch_size \
#                                         --save_exist \
#                                         --train_val_test_nums $train_val_test_nums > ./task_evaluation_logs/${dataset_name}/knn_${model_dir}_${ckp_name}.txt 2>&1"
#                 # 此处等待程序启动
#                 sleep 3
#             fi
#             # 此处等待
#             sleep 1
#         done
#     done
# done

# Logistic Regression评测
# for dataset_name in "${dataset_names[@]}"
# do
#     echo "[开始评测每一折数据] ${dataset_name}"
#     max_jobs=30
#     gpucores="30"
#     gpumem="30k"
#     batch_size="128"

#     # 每一折针对每个数据集测试
#     echo "[开始每个模型测试] ${dataset_name}"
#     for model_dir in "${model_dirs[@]}"
#     do
#         # 如果是sam类的方法需要调整参数
#         if [ "${model_dir}" == "SAM" -o "${model_dir}" == "MedSAM" -o "${model_dir}" == "SAM_Med2d" ]; then
#             batch_size="16"
#             gpucores="100"
#             gpumem="81k"
#         fi

#         for ckp_dir in $base_dir/Model_Outputs/dinov2/$model_dir/eval/*/
#         do
#             # 统计当前有几个正在运行的jobs
#             running_count=$(baizectl job ls -t PYTORCH | awk '$1 ~ /^task/ && $3 == "RUNNING"' | wc -l)

#             # 如果当前正在运行的job数量大于限定值，则会等待，否则启动新的job
#             while [ "$running_count" -ge "$max_jobs" ]; do
#                 echo "waiting..."
#                 sleep 10
#                 running_count=$(baizectl job ls -t PYTORCH | awk '$1 ~ /^task/ && $3 == "RUNNING"' | wc -l)
#             done

#             # 提交当前job
#             ckp_name=$(basename "$ckp_dir")
#             echo "[LogisticR Evaluation] Now is processing model_dir: $model_dir/$ckp_name"
            
#             # 判断当前是否已经存在该模型的评测结果
#             FILE="$base_dir/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/logreg_$dataset_name/results_eval_logreg.json"
#             if [ -f "$FILE" ]; then
#                 # 如果文件存在
#                 echo "发现当前文件存在！跳过重新评测！"
#             else
#                 # 如果文件不存在
#                 echo "发现当前文件不存在，执行评测！"
#                 current_time=$(date +"%Y%m%d%H%M%S")
#                 job_name="task-logr-ana"
#                 job_name="${job_name//_/-}"
#                 job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
#                 job_name="${job_name}-${current_time}"

#                 # 创建log保存目录
#                 mkdir -p ./task_evaluation_logs/${dataset_name}

#                 baizectl job submit --shm-size 10240 \
#                             --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan15 \
#                             --name $job_name \
#                             --max-retries 1 \
#                             --workers 1 \
#                             --requests-resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
#                             --resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
#                             -- bash -c "HF_HOME='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint' HF_HUB_OFFLINE='1'\
#                                         PYTHONPATH=. python dinov2/eval/log_regression_new.py \
#                                         --config-file $base_dir/Model_Outputs/dinov2/$model_dir/config.yaml \
#                                         --pretrained-weights $base_dir/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/teacher_checkpoint.pth \
#                                         --output-dir $base_dir/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/logreg_$dataset_name \
#                                         --train-dataset ImageNet:split=TRAIN:root=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name:extra=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name/Extra \
#                                         --val-dataset ImageNet:split=VAL:root=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name:extra=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name/Extra \
#                                         --batch_size $batch_size \
#                                         --save_exist \
#                                         --train_val_test_nums $train_val_test_nums > ./task_evaluation_logs/${dataset_name}/logr_${model_dir}_${ckp_name}.txt 2>&1"
                
#                 # 此处等待程序启动
#                 sleep 3
#             fi
#             # 此处等待
#             sleep 1
#         done
#     done
# done

# Linear Regression评测
## 评测linear时使用的模型
model_dirs=("Resnet50" "Clip" "Clip" "Clip" 
            "SAM" "SAM" "SAM" 
            "Dinov2_Official_Model_vitb14"
            "Dinov2_Official_Model_vitl14"
            "Dinov2_Official_Model_vitg14"
            "Biomedclip" "SAM_Med2d" "LVM_Resnet50" "LVM_vitb16")
ckp_dirs=("training_0001" "training_vitb16" "training_vitb32" "training_vitl14" 
          "training_vitb" "training_vith" "training_vitl" 
          "training_0001" 
          "training_0001" 
          "training_0001" 
          "training_vitb16" "training_0001" "training_0001" "training_0001")

for dataset_name in "${dataset_names[@]}"
do
    echo "[开始评测每一折数据] ${dataset_name}"
    max_jobs=30
    batch_size="32"
    gpucores="100"
    gpumem="81k"

    # 每一折针对每个数据集测试
    echo "[开始每个模型测试] ${dataset_name}"
    # for model_dir in "${model_dirs[@]}"
    for i in $(seq 0 $((${#model_dirs[@]} - 1)))
    do
        model_dir=${model_dirs[$i]}
        # 如果是sam类的方法需要调整参数
        if [ "${model_dir}" == "SAM" -o "${model_dir}" == "MedSAM" ]; then
            batch_size="10"
            gpucores="100"
            gpumem="81k"
        fi
        ckp_dir="${base_dir}/Model_Outputs/dinov2/${model_dir}/eval/${ckp_dirs[$i]}/"

        # for ckp_dir in $base_dir/Model_Outputs/dinov2/$model_dir/eval/*/
        # do
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
        echo "[Linear Evaluation] Now is processing model_dir: $model_dir/$ckp_name"

        # 判断当前是否已经存在该模型的评测结果
        FILE="$base_dir/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/linear_$dataset_name/results_eval_linear.json"
        if [ -f "$FILE" ]; then
            # 如果文件存在
            echo "发现当前文件存在！跳过重新评测！"
        else
            # 如果文件不存在
            echo "发现当前文件不存在，执行评测！"
            current_time=$(date +"%Y%m%d%H%M%S")
            job_name="task-logr-ana"
            job_name="${job_name//_/-}"
            job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
            job_name="${job_name}-${current_time}"

            # 创建log保存目录
            mkdir -p ./task_evaluation_logs/${dataset_name}
            
            baizectl job submit --shm-size 10240 \
                        --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan15 \
                        --name $job_name \
                        --max-retries 1 \
                        --workers 1 \
                        --requests-resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                        --resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                        -- bash -c "HF_HOME='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint' HF_HUB_OFFLINE='1'\
                                    PYTHONPATH=. python dinov2/eval/linear_new.py \
                                    --config-file $base_dir/Model_Outputs/dinov2/$model_dir/config.yaml \
                                    --pretrained-weights $base_dir/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/teacher_checkpoint.pth \
                                    --output-dir $base_dir/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/linear_$dataset_name \
                                    --train-dataset ImageNet:split=TRAIN:root=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name:extra=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name/Extra \
                                    --val-dataset ImageNet:split=VAL:root=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name:extra=$base_dir/Origin_Datasets/9h_fm_downstream_tasks/anatomical_type_classification/$dataset_name/Extra \
                                    --batch_size $batch_size \
                                    --save_exist \
                                    --train_val_test_nums $train_val_test_nums > ./task_evaluation_logs/${dataset_name}/linear_${model_dir}_${ckp_name}.txt 2>&1"
            # 此处等待程序启动
            sleep 3
        fi
        # 此处等待
        sleep 1
        # done
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