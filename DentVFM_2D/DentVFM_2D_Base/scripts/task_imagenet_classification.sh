# 评测下游任务：imagenet-1k分类任务，测试原始数据效果
dataset_name="ILSVRC_2012"
# model_dirs=("ILSVRC_2012_Test0001")
model_dirs=("Dinov2_Official_Model_vitl14")
device_name=0
# ckp_name="iter"           # 自动测试当前目录下所有ckp
ckp_name='training_0001'    # 指定某个ckp

# server
for model_dir in "${model_dirs[@]}"
do 
    if [ "$ckp_name" = "iter" ]; then
        echo "ckp_name=iter: $ckp_name"
        for ckp_dir in /home/jovyan/dataset/Model_Outputs/dinov2/$model_dir/eval/*/
        do
            ckp_name=$(basename "$ckp_dir")
            echo "Now is processing model_dir: $model_dir/$ckp_name"
            CUDA_VISIBLE_DEVICES=$device_name PYTHONPATH=. python dinov2/eval/knn.py \
                --config-file /home/jovyan/dataset/Model_Outputs/dinov2/$model_dir/config.yaml \
                --pretrained-weights /home/jovyan/dataset/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/teacher_checkpoint.pth \
                --output-dir /home/jovyan/dataset/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/knn_$dataset_name \
                --train-dataset ImageNet:split=TRAIN:root=/home/jovyan/dataset/Origin_Datasets/$dataset_name/Data/CLS-LOC:extra=/home/jovyan/dataset/Origin_Datasets/$dataset_name/extra \
                --val-dataset ImageNet:split=VAL:root=/home/jovyan/dataset/Origin_Datasets/$dataset_name/Data/CLS-LOC:extra=/home/jovyan/dataset/Origin_Datasets/$dataset_name/extra \
                --nb_knn 10 15 20 \
                --batch-size 1024 \
                --train_val_test_nums 1281167 50000 100000
        done
    else
        echo "ckp_name!=iter: $ckp_name"
        echo "Now is processing model_dir: $model_dir/$ckp_name"
        CUDA_VISIBLE_DEVICES=$device_name PYTHONPATH=. python dinov2/eval/knn.py \
            --config-file /home/jovyan/dataset/Model_Outputs/dinov2/$model_dir/config.yaml \
            --pretrained-weights /home/jovyan/dataset/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/teacher_checkpoint.pth \
            --output-dir /home/jovyan/dataset/Model_Outputs/dinov2/$model_dir/eval/$ckp_name/knn_$dataset_name \
            --train-dataset ImageNet:split=TRAIN:root=/home/jovyan/dataset/Origin_Datasets/$dataset_name/Data/CLS-LOC:extra=/home/jovyan/dataset/Origin_Datasets/$dataset_name/extra \
            --val-dataset ImageNet:split=VAL:root=/home/jovyan/dataset/Origin_Datasets/$dataset_name/Data/CLS-LOC:extra=/home/jovyan/dataset/Origin_Datasets/$dataset_name/extra \
            --nb_knn 10 15 20 \
            --batch-size 1024 \
            --train_val_test_nums 1281167 50000 100000
    fi
done