# 评测 M2F 框架下的模型
# 评测下游任务：分割任务
# 适合多数据集，多fold，多模型评测
# 该脚本适用服务器集群多卡并行评测

import time
import os
from datetime import datetime

# 数据集相关参数
kfold = 4
base_dir = "/home/jovyan/dataset"
re_run = False

# 要测试的数据集参数配置
mmseg_dir="/home/jovyan/dataset/Code/Project_Foundation_Model/dinov2_2d_version/dinov2/eval/evaluate_2d_segmentation"
model_dir="/home/jovyan/dataset/Model_Outputs/dinov2"

dataset_type_name_list = [
    # {"dataset_type": "miccai23sts2d",
    #  "data_name":"miccai23sts2d_segmentation_v1"},
    {"dataset_type": "dc100",
     "data_name":"DC1000_segmentation_v1"},
    # {"dataset_type": "duallabelpan",
    #  "data_name":"A_dual_labeled_dataset_segmentation_v1"}
]

model_ckp_config_list = [
    {
        "model_name": 'unet',
        "ckp": ['training_0001'],
        'config_file': 'unet-s5-d16_fcn_4xb4-20k_miccai-128x128.py'
    }
]


# 相关参数
max_jobs=30
gpucores="100"
gpumem="81k"


print("下游任务共：",len(dataset_type_name_list), "个", dataset_type_name_list)
print("[逐个任务开始评测：]")


# 更新ckp名称
for model_ckp_config in model_ckp_config_list:
    # 当前模型要测试的ckp
    ckp_dir = model_ckp_config["ckp"]
    model_name = model_ckp_config["model_name"]
    if ckp_dir[0] == "all" and len(ckp_dir) == 1:
        # 如果ckp_dir是all，则获取当前模型下所有的ckp
        ckp_dir=os.listdir(f"{base_dir}/Model_Outputs/dinov2/{model_name}/eval")
        # 排除.ipynb_checkpoints
        ckp_dir = [ckp for ckp in ckp_dir if not ckp.startswith(".ipynb_checkpoints")]
        # 更新ckp
        model_ckp_config["ckp"] = ckp_dir


# 评测seg
print("[开始评测分割任务]")
for dataset_type_name in dataset_type_name_list:
    dataset_type=dataset_type_name['dataset_type']
    data_name=dataset_type_name['data_name']

    print("当前正在测评的数据集是：", dataset_type, data_name)

    # 遍历kfold
    print("[针对kfold进行评测]")
    for fold_id in range(kfold-1, kfold):

        data_root = f"{data_name}_fold{fold_id+1}"
        print("当前kfold评测数据集为: ", data_root)

        # 遍历每类方法
        print("[每个目标模型都要在该数据集上进行测试！！！]")
        for model_ckp_config in model_ckp_config_list:
            model_name=model_ckp_config["model_name"]
            ckp_names=model_ckp_config["ckp"]
            config_file=model_ckp_config["config_file"]

            # 每类方法有多个ckp时，处理每个ckp
            for ckp in ckp_names:
                print("[M2F Seg Evaluation] Now is processing model_dir: ", model_name, ckp)

                # 运行命令
                now = datetime.now()
                current_time = now.strftime("%Y%m%d%H%M%S")
                job_name=f"task-segm2f-{dataset_type}-{current_time}"

                sys_cmd = f"cd {mmseg_dir};\
                            baizectl job submit --shm-size 102400 \
                            --image 10.1.17.1/9h/dinov2:5.3 \
                            --name {job_name} \
                            --restart-policy never \
                            --workers 1 \
                            --requests-resources cpu=24,nvidia.com/gpucores={gpucores},nvidia.com/gpumem={gpumem},nvidia.com/vgpu=1 \
                            --resources cpu=24,nvidia.com/gpucores={gpucores},nvidia.com/gpumem={gpumem},nvidia.com/vgpu=1 \
                            -- bash -c 'HF_HOME=/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint HF_HUB_OFFLINE=1\
                                        HF_HUB_CACHE=/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint/hub\
                                        /opt/conda/envs/dinov2/bin/python inference.py \
                                        ./configs/{dataset_type}/{config_file} \
                                        {model_dir}/{model_name}/eval/{ckp}/segm2f_{data_root}/best_mIoU_iter_4400.pth \
                                        --data_root {data_root} \
                                        --work-dir {model_dir}/{model_name}/eval/{ckp}/segm2f_{data_root}/inference/ \
                                        --out {model_dir}/{model_name}/eval/{ckp}/segm2f_{data_root}/inference/out'"
                
                os.system(sys_cmd)
                # 此处等待程序启动
                time.sleep(2)
print("[分割任务评测完成！]")