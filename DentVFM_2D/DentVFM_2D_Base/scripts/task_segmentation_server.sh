# 2d 分割任务测试脚本 mmseg+BNHead

# 分割评测路径
# mmseg_dir="/home/huangxinrui/Code/Project_Foundation_Model/dinov2_2d_version/dinov2/eval/evaluate_2d_segmentation"
mmseg_dir="/home/jovyan/dataset/Code/Project_Foundation_Model/dinov2_2d_version/dinov2/eval/evaluate_2d_segmentation"

# 模型保存路径
# model_dir="/mnt/ugreenf8c5_samba_share/huangxinrui/Model_Outputs/dinov2"
model_dir="/home/jovyan/dataset/Model_Outputs/dinov2"

# 要测试的数据集路径
# dataset_types=("miccai23sts2d")
# data_names=("miccai23sts2d_segmentation_v1")
# dataset_types=("childrenpan")
# data_names=("Children_dental_caries_segmentation_v1")
# dataset_types=("panmandibles")
# data_names=("Panoramic_Xrays_With_Segmented_Mandibles_segmentation_v1")
dataset_types=("turftsdentaldata")
data_names=("Tufts_Dental_Database_segmentation_v1")
# dataset_types=("dc100")
# data_names=("DC1000_segmentation_v1")

# dataset_types=("miccai23sts2d" "childrenpan" "panmandibles" 
#                "turftsdentaldata" "dc100")
# data_names=("miccai23sts2d_segmentation_v1" "Children_dental_caries_segmentation_v1" "Panoramic_Xrays_With_Segmented_Mandibles_segmentation_v1" 
#             "Tufts_Dental_Database_segmentation_v1" "DC1000_segmentation_v1")


cd "${mmseg_dir}"
for i in $(seq 0 $((${#dataset_types[@]} - 1)))
do  
    dataset_type=${dataset_types[$i]}
    data_name=${data_names[$i]}
    gpucores="100"
    gpumem="81k"
    # 数据路径
    # data_root="/mnt/ugreenf8c5_samba_share/huangxinrui/Origin_Datasets/9h_fm_downstream_tasks/miccai23sts2d_segmentation/${data_name}"
    # data_root="/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/miccai23sts2d_segmentation/${data_name}"

    # sam类方法
    ## SAM_vitb
    model_name="SAM_vitb"
    ckp_name="training_0001"
    current_time=$(date +"%Y%m%d%H%M%S")
    job_name="task-seg-${model_name}"
    job_name="${job_name//_/-}"
    job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
    job_name="${job_name}-${current_time}"
    baizectl job submit --shm-size 10240 \
                --image 10.1.17.1/9h/dinov2:5.1 \
                --name $job_name \
                --restart-policy never \
                --workers 1 \
                --requests-resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                --resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                -- bash -c "HF_HOME='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint' HF_HUB_OFFLINE='1'\
                            /opt/conda/envs/dinov2/bin/python train.py \
                            ./configs/${dataset_type}/sam_vitb_config.py \
                            --amp \
                            --work-dir ${model_dir}/${model_name}/eval/${ckp_name}/seg_${data_name} \
                            >../../../task_evaluation_logs/${data_name}/seg_${model_name}.txt 2>&1"
    sleep 5
    

    ## SAM_vith
    model_name="SAM_vith"
    ckp_name="training_0001"
    current_time=$(date +"%Y%m%d%H%M%S")
    job_name="task-seg-${model_name}"
    job_name="${job_name//_/-}"
    job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
    job_name="${job_name}-${current_time}"
    baizectl job submit --shm-size 10240 \
                --image 10.1.17.1/9h/dinov2:5.1 \
                --name $job_name \
                --restart-policy never \
                --workers 1 \
                --requests-resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                --resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                -- bash -c "HF_HOME='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint' HF_HUB_OFFLINE='1'\
                            /opt/conda/envs/dinov2/bin/python train.py \
                            ./configs/${dataset_type}/sam_vith_config.py \
                            --amp \
                            --work-dir ${model_dir}/${model_name}/eval/${ckp_name}/seg_${data_name} \
                            >../../../task_evaluation_logs/${data_name}/seg_${model_name}.txt 2>&1"
    sleep 5

    
    ## SAM_vitl
    model_name="SAM_vitl"
    ckp_name="training_0001"
    current_time=$(date +"%Y%m%d%H%M%S")
    job_name="task-seg-${model_name}"
    job_name="${job_name//_/-}"
    job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
    job_name="${job_name}-${current_time}"
    baizectl job submit --shm-size 10240 \
                --image 10.1.17.1/9h/dinov2:5.1 \
                --name $job_name \
                --restart-policy never \
                --workers 1 \
                --requests-resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                --resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                -- bash -c "HF_HOME='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint' HF_HUB_OFFLINE='1'\
                            /opt/conda/envs/dinov2/bin/python train.py \
                            ./configs/${dataset_type}/sam_vitl_config.py \
                            --amp \
                            --work-dir ${model_dir}/${model_name}/eval/${ckp_name}/seg_${data_name} \
                            >../../../task_evaluation_logs/${data_name}/seg_${model_name}.txt 2>&1"
    sleep 5


    ## SAMMed2d
    model_name="SAM_Med2d"
    ckp_name="training_0001"
    current_time=$(date +"%Y%m%d%H%M%S")
    job_name="task-seg-${model_name}"
    job_name="${job_name//_/-}"
    job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
    job_name="${job_name}-${current_time}"
    baizectl job submit --shm-size 10240 \
                --image 10.1.17.1/9h/dinov2:5.1 \
                --name $job_name \
                --restart-policy never \
                --workers 1 \
                --requests-resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                --resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                -- bash -c "HF_HOME='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint' HF_HUB_OFFLINE='1'\
                            /opt/conda/envs/dinov2/bin/python train.py \
                            ./configs/${dataset_type}/sammed_vitb_config.py \
                            --amp \
                            --work-dir ${model_dir}/${model_name}/eval/${ckp_name}/seg_${data_name} \
                            >../../../task_evaluation_logs/${data_name}/seg_${model_name}.txt 2>&1"
    sleep 5

    
    # resnet50方法
    # model_name="Resnet50"
    # ckp_name="training_0001"
    # CUDA_VISIBLE_DEVICES=3 python train.py ./configs/resnet50_${dataset_type}_config.py \
    #                                        --amp \
    #                                        --work-dir ${model_dir}/${model_name}/eval/${ckp_name}/seg_${data_name} \
    #                                        >../../../task_evaluation_logs/${data_name}/seg_${model_name}.txt 2>&1 &

    # clip类方法
    ## clip_vitb16
    model_name="Clip_vitb16"
    ckp_name="training_0001"
    current_time=$(date +"%Y%m%d%H%M%S")
    job_name="task-seg-${model_name}"
    job_name="${job_name//_/-}"
    job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
    job_name="${job_name}-${current_time}"
    baizectl job submit --shm-size 10240 \
                --image 10.1.17.1/9h/dinov2:5.1 \
                --name $job_name \
                --restart-policy never \
                --workers 1 \
                --requests-resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                --resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                -- bash -c "HF_HOME='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint' HF_HUB_OFFLINE='1'\
                            /opt/conda/envs/dinov2/bin/python train.py \
                            ./configs/${dataset_type}/clip_vitb16_config.py \
                            --amp \
                            --work-dir ${model_dir}/${model_name}/eval/${ckp_name}/seg_${data_name} \
                            >../../../task_evaluation_logs/${data_name}/seg_${model_name}.txt 2>&1"
    sleep 5


    ## clip_vitb32
    model_name="Clip_vitb32"
    ckp_name="training_0001"
    current_time=$(date +"%Y%m%d%H%M%S")
    job_name="task-seg-${model_name}"
    job_name="${job_name//_/-}"
    job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
    job_name="${job_name}-${current_time}"
    baizectl job submit --shm-size 10240 \
                --image 10.1.17.1/9h/dinov2:5.1 \
                --name $job_name \
                --restart-policy never \
                --workers 1 \
                --requests-resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                --resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                -- bash -c "HF_HOME='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint' HF_HUB_OFFLINE='1'\
                            /opt/conda/envs/dinov2/bin/python train.py \
                            ./configs/${dataset_type}/clip_vitb32_config.py \
                            --amp \
                            --work-dir ${model_dir}/${model_name}/eval/${ckp_name}/seg_${data_name} \
                            >../../../task_evaluation_logs/${data_name}/seg_${model_name}.txt 2>&1"
    sleep 5


    ## clip_vitl14
    model_name="Clip_vitl14"
    ckp_name="training_0001"
    current_time=$(date +"%Y%m%d%H%M%S")
    job_name="task-seg-${model_name}"
    job_name="${job_name//_/-}"
    job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
    job_name="${job_name}-${current_time}"
    baizectl job submit --shm-size 10240 \
                --image 10.1.17.1/9h/dinov2:5.1 \
                --name $job_name \
                --restart-policy never \
                --workers 1 \
                --requests-resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                --resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                -- bash -c "HF_HOME='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint' HF_HUB_OFFLINE='1'\
                            /opt/conda/envs/dinov2/bin/python train.py \
                            ./configs/${dataset_type}/clip_vitl14_config.py \
                            --amp \
                            --work-dir ${model_dir}/${model_name}/eval/${ckp_name}/seg_${data_name} \
                            >../../../task_evaluation_logs/${data_name}/seg_${model_name}.txt 2>&1"
    sleep 5


    ## biomedclip_vitb16
    model_name="Biomedclip"
    ckp_name="training_vitb16"
    current_time=$(date +"%Y%m%d%H%M%S")
    job_name="task-seg-${model_name}"
    job_name="${job_name//_/-}"
    job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
    job_name="${job_name}-${current_time}"
    baizectl job submit --shm-size 10240 \
                --image 10.1.17.1/9h/dinov2:5.1 \
                --name $job_name \
                --restart-policy never \
                --workers 1 \
                --requests-resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                --resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                -- bash -c "HF_HOME='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint' HF_HUB_OFFLINE='1'\
                            HF_HUB_CACHE='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint/hub'\
                            /opt/conda/envs/dinov2/bin/python train.py \
                            ./configs/${dataset_type}/biomedclip_vitb16_config.py \
                            --amp \
                            --work-dir ${model_dir}/${model_name}/eval/${ckp_name}/seg_${data_name} \
                            >../../../task_evaluation_logs/${data_name}/seg_${model_name}.txt 2>&1"
    sleep 5


    # LVMMED
    model_name="LVM_vitb16"
    ckp_name="training_0001"
    current_time=$(date +"%Y%m%d%H%M%S")
    job_name="task-seg-${model_name}"
    job_name="${job_name//_/-}"
    job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
    job_name="${job_name}-${current_time}"
    baizectl job submit --shm-size 10240 \
                --image 10.1.17.1/9h/dinov2:5.1 \
                --name $job_name \
                --restart-policy never \
                --workers 1 \
                --requests-resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                --resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
                -- bash -c "HF_HOME='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint' HF_HUB_OFFLINE='1'\
                            /opt/conda/envs/dinov2/bin/python train.py \
                            ./configs/${dataset_type}/lvmmed_vitb16_config.py \
                            --amp \
                            --work-dir ${model_dir}/${model_name}/eval/${ckp_name}/seg_${data_name} \
                            >../../../task_evaluation_logs/${data_name}/seg_${model_name}.txt 2>&1"
    sleep 5


    # ours 类dinov2的模型
    # model_name="Dinov2_Official_Model_vitb14"
    # ckp_name="training_0001"
    # current_time=$(date +"%Y%m%d%H%M%S")
    # job_name="task-seg-${model_name}"
    # job_name="${job_name//_/-}"
    # job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
    # job_name="${job_name}-${current_time}"
    # baizectl job submit --shm-size 10240 \
    #             --image 10.1.17.1/9h/dinov2:5.1 \
    #             --name $job_name \
    #             --restart-policy never \
    #             --workers 1 \
    #             --requests-resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
    #             --resources nvidia.com/gpucores=$gpucores,nvidia.com/gpumem=$gpumem,nvidia.com/vgpu=1 \
    #             -- bash -c "HF_HOME='/home/jovyan/dataset/Code/Project_Foundation_Model/biomedclip/checkpoint' HF_HUB_OFFLINE='1'\
    #                         /opt/conda/envs/dinov2/bin/python train.py \
    #                         ./configs/dinov2_vitb14_${dataset_type}_config.py \
    #                         --amp \
    #                         --work-dir ${model_dir}/${model_name}/eval/${ckp_name}/seg_${data_name} \
    #                         >../../../task_evaluation_logs/${data_name}/seg_${model_name}.txt 2>&1"

done