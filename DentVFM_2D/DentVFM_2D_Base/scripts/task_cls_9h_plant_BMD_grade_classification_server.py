# 评测下游任务：BMD分类任务
# 适合多数据集，多fold，多模型评测
# 该脚本适用服务器集群多卡并行评测

import time
import os
from datetime import datetime
import random

# 数据集相关参数
kfold = 1
base_dir = "/home/jovyan/dataset"
re_run = False

# 要测试的数据集信息
base_dataset_names = ["9h_BMD_grade_crop_dilation_20_v2",
                      "9h_BMD_grade_crop_dilation_20_4class_v2",
                      "9h_BMD_grade_crop_dilation_100_v2",
                      "9h_BMD_grade_crop_dilation_100_4class_v2"]
# base_dataset_names=["9h_BMD_grade_crop_dilation_20_v1",
#                     "9h_age_classification_lateral_v1",
#                     "9h_age_classification_panoramic_v1", 
#                     "9h_cyst_tumor_cls_4type_pan_v1",    
#                     "9h_fracture_type_classification_xray_pan_v1",  
#                     "9h_periodontal_grade_classification_v1",   
#                     "9h_single_double_after_xray_v1",   
#                     "9h_single_double_before_xray_v1",   
#                     "anatomical_type_classification_ANB_v2",  
#                     "apicoectomy_surgery_panoramic_cls_v2",    
#                     "teeth_patch_diagnosis_cls_DRAD_Kaggle_v1",  
#                     "teeth_patch_diagnosis_cls_miccai23_DENTEX_task3_v1"]
dataset_dirs = ["9h_plant_BMD_grade_classification",
                "9h_plant_BMD_grade_classification",
                "9h_plant_BMD_grade_classification",
                "9h_plant_BMD_grade_classification"]
# dataset_dirs = ["9h_plant_BMD_grade_classification",
#                 "9h_age_classification",
#                 "9h_age_classification",
#                 "9h_cyst_type_classification",
#                 "9h_fracture_type_classification",
#                 "9h_periodontal_grade_classification",
#                 "9h_operation_type_classification",
#                 "9h_operation_type_classification",
#                 "anatomical_type_classification",
#                 "apicoectomy_surgery_panoramic",
#                 "teeth_patch_diagnosis_classification",
#                 "teeth_patch_diagnosis_classification"]
short_names = ["bmd20",
               "bmd204",
               "bmd100",
               "bmd1004"]
# short_names = ["bmd",
#                "agelat",
#                "agepan",
#                "cyst",
#                "frac",
#                "peri",
#                "opaf",
#                "opbe",
#                "ana",
#                "api",
#                "drad",
#                "dentex"]
train_val_test_nums = ["1044 269 269",
                       "1108 267 267",
                       "1044 269 269",
                       "1108 267 267"]
# train_val_test_nums = ["618 269 269",
#                        "1209 521 521",
#                        "1029 457 457",
#                        "422 184 184",
#                        "379 165 165",
#                        "602 260 260",
#                        "752 325 325",
#                        "207 90 90",
#                        "167 169 169",
#                        "68 68 68",
#                        "1392 600 600",
#                        "440 192 192"]


# 要对比的相关模型
## VFM对比方法
# model_dirs=["Fm_2d_Test0008-1",
#             "Fm_2d_Test0008-2",
#             "Fm_2d_Test0010-2",
#             "Fm_2d_Test0010-3",
#             "Fm_2d_Test0011-1"]
# ckp_dirs=[["all"],
#           ["all"],
#           ["all"],
#           ["all"],
#           ["all"]]

model_dirs=["Resnet50",
            "Clip", 
            "SAM", 
            "Dinov2_Official_Model_vitb14", 
            "Dinov2_Official_Model_vitl14",
            "Dinov2_Official_Model_vitg14", 
            "Biomedclip", 
            "SAM_Med2d", 
            "LVM_Resnet50", 
            "LVM_vitb16",
            "Fm_2d_Test0008-3", 
            "Fm_2d_Test0010-2",
            "Fm_2d_Test0011-1",
            "MAE_2d_9h_vitb14",
            "MAE_2d_9h_vitg14",
            "MAE_2d_9h_vitl14",
            "iBoT_2d_9h_vitb14",
            "iBoT_2d_9h_vitl14"]
ckp_dirs=[["all"],
          ['all'],
          ['all'],
          ['all'],
          ['all'],
          ['all'],
          ['all'],
          ['all'],
          ['all'],
          ['all'],
          ["training_137499"],
          ["training_612499"],
          ["training_99999"],
          ["training_0520"],
          ["training_0140"],
          ["training_0260"],
          ["training_0600"],
          ["training_0760"]]

# 相关参数
max_jobs=30
gpucores="30"
gpumem="30k"
batch_size="128"


print("下游任务共：",len(base_dataset_names), "个", base_dataset_names)
print("逐个任务开始评测：")


# 更新ckp名称
for model_id, model_dir in enumerate(model_dirs):
    # 当前模型要测试的ckp
    ckp_dir = ckp_dirs[model_id]
    if ckp_dir[0] == "all" and len(ckp_dir) == 1:
        # 如果ckp_dir是all，则获取当前模型下所有的ckp
        ckp_dir=os.listdir(f"{base_dir}/Model_Outputs/dinov2/{model_dir}/eval")
        # 排除.ipynb_checkpoints
        ckp_dir = [ckp for ckp in ckp_dir if not ckp.startswith(".ipynb_checkpoints")]
        # 更新ckp
        ckp_dirs[model_id] = ckp_dir


# 评测knn
print("开始评测KNN：")
for idx,base_dataset_name in enumerate(base_dataset_names):

    # 数据集简称
    short_name = short_names[idx]

    # 数据集前序路径
    dataset_dir = dataset_dirs[idx]

    # 当前数据集的划分
    train_val_test_num = train_val_test_nums[idx]

    # kfold个子数据集
    dataset_names=[]
    for fold_id in range(kfold):
        dataset_names.append(f"{base_dataset_name}_fold{fold_id+1}")

    # 可视化相关信息
    print(f"[当前正在评测数据集] {base_dataset_name}, 包含fold数据集: {dataset_names}")
    print(f"[当前评测数据集的划分] {train_val_test_num}")
    

    # 遍历每一折数据评测
    # KNN评测
    for dataset_name in dataset_names:
        print(f"[开始评测每一折数据] {dataset_name}")
        
        # 遍历每一个要测试的模型
        for model_id, model_dir in enumerate(model_dirs):
            
            # 当前模型要测试的ckp
            ckp_dir = ckp_dirs[model_id]

            # 如果是sam类的方法需要调整参数
            if model_dir == "SAM" or model_dir == "MedSAM" or model_dir == "SAM_Med2d":
                batch_size="16"
                gpucores="100"
                gpumem="81k"

            # if ckp_dir[0] == "all" and len(ckp_dir) == 1:
            #     # 如果ckp_dir是all，则获取当前模型下所有的ckp
            #     ckp_dir=os.listdir(f"{base_dir}/Model_Outputs/dinov2/{model_dir}/eval")
            
            # 遍历每个ckp都进行测试
            for ckp_name in ckp_dir:
                print(f"[KNN Evaluation] Now is processing model_dir: {model_dir}/{ckp_name}")

                # 判断当前是否已经存在该模型的评测结果
                if os.path.exists(f"{base_dir}/Model_Outputs/dinov2/{model_dir}/eval/{ckp_name}/knn_{dataset_name}/results_eval_knn.json") and re_run == False:
                    # 如果文件存在
                    print("发现当前文件存在！跳过重新评测！")
                    continue
                else:
                    # 如果文件不存在
                    print("发现当前文件不存在或者需要强制重新运算，执行评测！")
                    now = datetime.now()
                    current_time = now.strftime("%Y%m%d%H%M%S")
                    job_name=f"task-knn-{short_name}-{current_time}"
                    
                    # 创建log保存目录
                    os.makedirs(f"./task_evaluation_logs/{dataset_name}", exist_ok=True)
                    
                    
                    sys_cmd = f"baizectl job submit --shm-size 10240 \
                                --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan15 \
                                --name {job_name} \
                                --restart-policy never \
                                --workers 1 \
                                --requests-resources nvidia.com/gpucores={gpucores},nvidia.com/gpumem={gpumem},nvidia.com/vgpu=1 \
                                --resources nvidia.com/gpucores={gpucores},nvidia.com/gpumem={gpumem},nvidia.com/vgpu=1 \
                                -- bash -c 'HF_HOME=/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint HF_HUB_OFFLINE=1\
                                            PYTHONPATH=. python dinov2/eval/knn_new.py \
                                            --config-file {base_dir}/Model_Outputs/dinov2/{model_dir}/config.yaml \
                                            --pretrained-weights {base_dir}/Model_Outputs/dinov2/{model_dir}/eval/{ckp_name}/teacher_checkpoint.pth \
                                            --output-dir {base_dir}/Model_Outputs/dinov2/{model_dir}/eval/{ckp_name}/knn_{dataset_name} \
                                            --train-dataset ImageNet:split=TRAIN:root={base_dir}/Origin_Datasets/9h_fm_downstream_tasks/{dataset_dir}/{dataset_name}:extra={base_dir}/Origin_Datasets/9h_fm_downstream_tasks/{dataset_dir}/{dataset_name}/Extra \
                                            --val-dataset ImageNet:split=VAL:root={base_dir}/Origin_Datasets/9h_fm_downstream_tasks/{dataset_dir}/{dataset_name}:extra={base_dir}/Origin_Datasets/9h_fm_downstream_tasks/{dataset_dir}/{dataset_name}/Extra \
                                            --nb_knn 10 15 20 \
                                            --batch_size {batch_size} \
                                            --train_val_test_nums {train_val_test_num} > ./task_evaluation_logs/{dataset_name}/knn_{model_dir}_{ckp_name}.txt 2>&1'"
                    
                    os.system(sys_cmd)
                    # 此处等待程序启动
                    time.sleep(1)

                    # 以一定概率删除现在已经成功运行完毕的任务
                    if random.random() < 0.1:
                        # 此处删除已经运行成功的程序，防止列表过多
                        print("删除已经成功运行完成的任务，防止任务过多！")
                        delete_cmd = "baizectl job ls -t PYTORCH | awk '$1 ~ /^task/ && $3 == \"SUCCEEDED\"  {print $1}' | xargs -I {} baizectl job delete {}"
                        os.system(delete_cmd)
                        time.sleep(1)


# 评测logreg
print("开始评测Logreg：")
for idx,base_dataset_name in enumerate(base_dataset_names):

    # 数据集简称
    short_name = short_names[idx]

    # 数据集前序路径
    dataset_dir = dataset_dirs[idx]

    # 当前数据集的划分
    train_val_test_num = train_val_test_nums[idx]

    # kfold个子数据集
    dataset_names=[]
    for fold_id in range(kfold):
        dataset_names.append(f"{base_dataset_name}_fold{fold_id+1}")

    # 可视化相关信息
    print(f"[当前正在评测数据集] {base_dataset_name}, 包含fold数据集: {dataset_names}")
    print(f"[当前评测数据集的划分] {train_val_test_num}")
    

    # 遍历每一折数据评测
    # KNN评测
    for dataset_name in dataset_names:
        print(f"[开始评测每一折数据] {dataset_name}")
        
        # 遍历每一个要测试的模型
        for model_id, model_dir in enumerate(model_dirs):
            
            # 当前模型要测试的ckp
            ckp_dir = ckp_dirs[model_id]

            # 如果是sam类的方法需要调整参数
            if model_dir == "SAM" or model_dir == "MedSAM" or model_dir == "SAM_Med2d":
                batch_size="16"
                gpucores="100"
                gpumem="81k"

            # if ckp_dir[0] == "all" and len(ckp_dir) == 1:
            #     # 如果ckp_dir是all，则获取当前模型下所有的ckp
            #     ckp_dir=os.listdir(f"{base_dir}/Model_Outputs/dinov2/{model_dir}/eval")
            
            # 遍历每个ckp都进行测试
            for ckp_name in ckp_dir:
                print(f"[LogisticR Evaluation] Now is processing model_dir: {model_dir}/{ckp_name}")

                # 判断当前是否已经存在该模型的评测结果
                if os.path.exists(f"{base_dir}/Model_Outputs/dinov2/{model_dir}/eval/{ckp_name}/logreg_{dataset_name}/results_eval_logreg.json") and re_run == False:
                    # 如果文件存在
                    print("发现当前文件存在！跳过重新评测！")
                    continue
                else:
                    # 如果文件不存在
                    print("发现当前文件不存在或者需要强制重新运算，执行评测！")
                    now = datetime.now()
                    current_time = now.strftime("%Y%m%d%H%M%S")
                    job_name=f"task-logr-{short_name}-{current_time}"
                    
                    # 创建log保存目录
                    os.makedirs(f"./task_evaluation_logs/{dataset_name}", exist_ok=True)
                    
                    sys_cmd = f"baizectl job submit --shm-size 10240 \
                                --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan15 \
                                --name {job_name} \
                                --restart-policy never \
                                --workers 1 \
                                --requests-resources nvidia.com/gpucores={gpucores},nvidia.com/gpumem={gpumem},nvidia.com/vgpu=1 \
                                --resources nvidia.com/gpucores={gpucores},nvidia.com/gpumem={gpumem},nvidia.com/vgpu=1 \
                                -- bash -c 'HF_HOME=/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint HF_HUB_OFFLINE=1\
                                            PYTHONPATH=. python dinov2/eval/log_regression_new.py \
                                            --config-file {base_dir}/Model_Outputs/dinov2/{model_dir}/config.yaml \
                                            --pretrained-weights {base_dir}/Model_Outputs/dinov2/{model_dir}/eval/{ckp_name}/teacher_checkpoint.pth \
                                            --output-dir {base_dir}/Model_Outputs/dinov2/{model_dir}/eval/{ckp_name}/logreg_{dataset_name} \
                                            --train-dataset ImageNet:split=TRAIN:root={base_dir}/Origin_Datasets/9h_fm_downstream_tasks/{dataset_dir}/{dataset_name}:extra={base_dir}/Origin_Datasets/9h_fm_downstream_tasks/{dataset_dir}/{dataset_name}/Extra \
                                            --val-dataset ImageNet:split=VAL:root={base_dir}/Origin_Datasets/9h_fm_downstream_tasks/{dataset_dir}/{dataset_name}:extra={base_dir}/Origin_Datasets/9h_fm_downstream_tasks/{dataset_dir}/{dataset_name}/Extra \
                                            --batch_size {batch_size} \
                                            --train_val_test_nums {train_val_test_num} > ./task_evaluation_logs/{dataset_name}/logr_{model_dir}_{ckp_name}.txt 2>&1'"
                    
                    os.system(sys_cmd)
                    # 此处等待程序启动
                    time.sleep(1)

                    # 以一定概率删除现在已经成功运行完毕的任务
                    if random.random() < 0.1:
                        # 此处删除已经运行成功的程序，防止列表过多
                        print("删除已经成功运行完成的任务，防止任务过多！")
                        delete_cmd = "baizectl job ls -t PYTORCH | awk '$1 ~ /^task/ && $3 == \"SUCCEEDED\"  {print $1}' | xargs -I {} baizectl job delete {}"
                        os.system(delete_cmd)
                        time.sleep(1)


# 评测linear
print("开始评测Linear：")
for idx,base_dataset_name in enumerate(base_dataset_names):

    # 数据集简称
    short_name = short_names[idx]

    # 数据集前序路径
    dataset_dir = dataset_dirs[idx]

    # 当前数据集的划分
    train_val_test_num = train_val_test_nums[idx]

    # kfold个子数据集
    dataset_names=[]
    for fold_id in range(kfold):
        dataset_names.append(f"{base_dataset_name}_fold{fold_id+1}")

    # 可视化相关信息
    print(f"[当前正在评测数据集] {base_dataset_name}, 包含fold数据集: {dataset_names}")
    print(f"[当前评测数据集的划分] {train_val_test_num}")
    

    # 遍历每一折数据评测
    # KNN评测
    for dataset_name in dataset_names:
        print(f"[开始评测每一折数据] {dataset_name}")
        
        # 遍历每一个要测试的模型
        for model_id, model_dir in enumerate(model_dirs):
            
            # 当前模型要测试的ckp
            ckp_dir = ckp_dirs[model_id]

            # 如果是sam类的方法需要调整参数
            if model_dir == "SAM" or model_dir == "MedSAM" or model_dir == "SAM_Med2d":
                batch_size="16"
                gpucores="100"
                gpumem="81k"

            # if ckp_dir[0] == "all" and len(ckp_dir) == 1:
            #     # 如果ckp_dir是all，则获取当前模型下所有的ckp
            #     ckp_dir=os.listdir(f"{base_dir}/Model_Outputs/dinov2/{model_dir}/eval")
            
            # 遍历每个ckp都进行测试
            for ckp_name in ckp_dir:
                print(f"[Linear Evaluation] Now is processing model_dir: {model_dir}/{ckp_name}")

                # 判断当前是否已经存在该模型的评测结果
                if os.path.exists(f"{base_dir}/Model_Outputs/dinov2/{model_dir}/eval/{ckp_name}/linear_{dataset_name}/results_eval_linear.json") and re_run == False:
                    # 如果文件存在
                    print("发现当前文件存在！跳过重新评测！")
                    continue
                else:
                    # 如果文件不存在
                    print("发现当前文件不存在或者需要强制重新运算，执行评测！")
                    now = datetime.now()
                    current_time = now.strftime("%Y%m%d%H%M%S")
                    job_name=f"task-linear-{short_name}-{current_time}"
                    
                    # 创建log保存目录
                    os.makedirs(f"./task_evaluation_logs/{dataset_name}", exist_ok=True)
                    
                    
                    sys_cmd = f"baizectl job submit --shm-size 10240 \
                                --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan15 \
                                --name {job_name} \
                                --restart-policy never \
                                --workers 1 \
                                --requests-resources nvidia.com/gpucores={gpucores},nvidia.com/gpumem={gpumem},nvidia.com/vgpu=1 \
                                --resources nvidia.com/gpucores={gpucores},nvidia.com/gpumem={gpumem},nvidia.com/vgpu=1 \
                                -- bash -c 'HF_HOME=/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint HF_HUB_OFFLINE=1\
                                            PYTHONPATH=. python dinov2/eval/linear_new.py \
                                            --config-file {base_dir}/Model_Outputs/dinov2/{model_dir}/config.yaml \
                                            --pretrained-weights {base_dir}/Model_Outputs/dinov2/{model_dir}/eval/{ckp_name}/teacher_checkpoint.pth \
                                            --output-dir {base_dir}/Model_Outputs/dinov2/{model_dir}/eval/{ckp_name}/linear_{dataset_name} \
                                            --train-dataset ImageNet:split=TRAIN:root={base_dir}/Origin_Datasets/9h_fm_downstream_tasks/{dataset_dir}/{dataset_name}:extra={base_dir}/Origin_Datasets/9h_fm_downstream_tasks/{dataset_dir}/{dataset_name}/Extra \
                                            --val-dataset ImageNet:split=VAL:root={base_dir}/Origin_Datasets/9h_fm_downstream_tasks/{dataset_dir}/{dataset_name}:extra={base_dir}/Origin_Datasets/9h_fm_downstream_tasks/{dataset_dir}/{dataset_name}/Extra \
                                            --batch_size {batch_size} \
                                            --train_val_test_nums {train_val_test_num} > ./task_evaluation_logs/{dataset_name}/linear_{model_dir}_{ckp_name}.txt 2>&1'"
                    
                    os.system(sys_cmd)
                    # 此处等待程序启动
                    time.sleep(1)

                    # 以一定概率删除现在已经成功运行完毕的任务
                    if random.random() < 0.1:
                        # 此处删除已经运行成功的程序，防止列表过多
                        print("删除已经成功运行完成的任务，防止任务过多！")
                        delete_cmd = "baizectl job ls -t PYTORCH | awk '$1 ~ /^task/ && $3 == \"SUCCEEDED\"  {print $1}' | xargs -I {} baizectl job delete {}"
                        os.system(delete_cmd)
                        time.sleep(1)