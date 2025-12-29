# 简化
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
# dataset_types=("turftsdentaldata")
# data_names=("Tufts_Dental_Database_segmentation_v1")
# dataset_types=("dc100")
# data_names=("DC1000_segmentation_v1")
# dataset_types=("duallabelpan")
# data_names=("A_dual_labeled_dataset_segmentation_v1")

dataset_types=("miccai23sts2d" "childrenpan" "panmandibles")
data_names=("miccai23sts2d_segmentation_v1" "Children_dental_caries_segmentation_v1" "Panoramic_Xrays_With_Segmented_Mandibles_segmentation_v1")


# 要测试的模型
model_names=(
             "SAM_vitb" 
             "SAM_vith"  
             "SAM_vitl" 
             "SAM_Med2d" 
             "Clip_vitb16" 
             "Clip_vitb32" 
             "Clip_vitl14" 
             "Biomedclip" 
             "LVM_vitb16")
ckp_names=(
           "training_0001" 
           "training_0001" 
           "training_0001" 
           "training_0001" 
           "training_0001" 
           "training_0001" 
           "training_0001" 
           "training_vitb16" 
           "training_0001")
config_files=(
              "sam_vitb_config.py" 
              "sam_vith_config.py" 
              "sam_vitl_config.py" 
              "sammed_vitb_config.py" 
              "clip_vitb16_config.py" 
              "clip_vitb32_config.py" 
              "clip_vitl14_config.py" 
              "biomedclip_vitb16_config.py" 
              "lvmmed_vitb16_config.py")

cd "${mmseg_dir}"
for i in $(seq 0 $((${#dataset_types[@]} - 1)))
do  
    dataset_type=${dataset_types[$i]}
    data_name=${data_names[$i]}
    gpucores="100"
    gpumem="81k"
    
    # 遍历每类方法
    for j in $(seq 0 $((${#model_names[@]} - 1)))
    do
        model_name=${model_names[$j]}
        ckp_name=${ckp_names[$j]}
        config_file=${config_files[$j]}

        current_time=$(date +"%Y%m%d%H%M%S")
        job_name="task-seg-${model_name}"
        job_name="${job_name//_/-}"
        job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
        job_name="${job_name}-${current_time}"
        
        # 创建log保存目录
        mkdir -p ../../../task_evaluation_logs/${data_name}
        
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
                                ./configs/${dataset_type}/${config_file} \
                                --amp \
                                --work-dir ${model_dir}/${model_name}/eval/${ckp_name}/seg_${data_name} \
                                >../../../task_evaluation_logs/${data_name}/seg_${model_name}.txt 2>&1"
        sleep 5
    done

done