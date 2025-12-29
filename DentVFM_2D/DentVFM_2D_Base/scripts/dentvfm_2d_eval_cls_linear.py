import time
import os
from datetime import datetime
import random
import subprocess
import sys

# dataset configurations
kfold = 5
base_dir = "XXXX"
base_dataset_names=[
                    # "9h_periodontal_FG_grade_classification",
                    # "9h_BMD_grade_crop_dilation_100_4class",
                    # "9h_age_classification_lateral",
                    # "9h_age_classification_panoramic", 
                    # "9h_cyst_tumor_cls_4type_pan",    
                    # "9h_fracture_type_classification_xray_pan",  
                    # "9h_periodontal_grade_classification",   
                    # "9h_single_double_after_xray",   
                    # "9h_single_double_before_xray",   
                    # "anatomical_type_classification_ANB",  
                    # "apicoectomy_surgery_panoramic_cls",    
                    # "teeth_patch_diagnosis_cls_DRAD_Kaggle",  
                    # "teeth_patch_diagnosis_cls_miccai23_DENTEX_task3",
                    # "DC1000_caries_grade_classification",
                    # "teeth_patch_diagnosis_cls_DXPD_Kaggle"
                   ]
dataset_dirs = [
                # "9h_periodontal_grade_classification",
                # "9h_plant_BMD_grade_classification",
                # "9h_age_classification",
                # "9h_age_classification",
                # "9h_cyst_type_classification",
                # "9h_fracture_type_classification",
                # "9h_periodontal_grade_classification",
                # "9h_operation_type_classification",
                # "9h_operation_type_classification",
                # "anatomical_type_classification",
                # "apicoectomy_surgery_panoramic",
                # "teeth_patch_diagnosis_classification",
                # "teeth_patch_diagnosis_classification",
                # "DC1000_caries_grade_classification",
                # "teeth_patch_diagnosis_classification"
               ]

# model configurations
model_dirs=["XXXX"]
ckp_dirs=[["all"]]
img_reses = [
    518,
]
for model_id, model_dir in enumerate(model_dirs):
    ckp_dir = ckp_dirs[model_id]
    if ckp_dir[0] == "all" and len(ckp_dir) == 1:
        ckp_dir=os.listdir(f"{base_dir}/output_dir/XXXX/{model_dir}/eval")
        ckp_dir = [ckp for ckp in ckp_dir if not ckp.startswith(".ipynb_checkpoints")]
        ckp_dirs[model_id] = ckp_dir

# evaluation configurations
re_run = False
batch_size="32"

# info
print("# of total tasks:",len(base_dataset_names), "details:", base_dataset_names)
print("start evaluate [Linear]:")

for idx,base_dataset_name in enumerate(base_dataset_names):
    dataset_dir = dataset_dirs[idx]

    dataset_names=[]
    for fold_id in range(kfold):
        dataset_names.append(f"{base_dataset_name}_fold{fold_id+1}")

    print(f"current dataset: {base_dataset_name}, all folds: {dataset_names}")

    for dataset_name in dataset_names:
        print(f"current fold: {dataset_name}")

        data_root = f'{base_dir}/{dataset_dir}/{dataset_name}'
        def count_jpeg_files(root_dir):
            count = 0
            for dirpath, dirnames, filenames in os.walk(root_dir):
                for filename in filenames:
                    if filename.lower().endswith(('.jpg', '.jpeg')):
                        count += 1
            return count
        train_num = count_jpeg_files(os.path.join(data_root, 'train'))
        val_num = count_jpeg_files(os.path.join(data_root, 'val'))
        train_val_test_num = f"{train_num} {val_num} {val_num}"
        print(f"current fole split: {train_val_test_num}")
        
        for model_id, model_dir in enumerate(model_dirs):
            ckp_dir = ckp_dirs[model_id]
            img_res = img_reses[model_id]
            

            for ckp_name in ckp_dir:
                print(f"[Linear Evaluation] Now is processing model_dir: {model_dir}/{ckp_name}")

                if os.path.exists(f"{base_dir}/output_dir/XXXX/{model_dir}/eval/{ckp_name}/linear_{dataset_name}/results_eval_linear.json") and re_run == False:
                    print("file exists! skip!")
                    continue
                else:
                    print("file not found, execute evaluation!")
                    sys_cmd = f"bash -c 'PYTHONPATH=. python dinov2/eval/linear_new.py \
                                        --config-file {base_dir}/output_dir/XXXX/{model_dir}/config.yaml \
                                        --pretrained-weights {base_dir}/output_dir/XXXX/{model_dir}/eval/{ckp_name}/teacher_checkpoint.pth \
                                        --output-dir {base_dir}/output_dir/XXXX/{model_dir}/eval/{ckp_name}/linear_{dataset_name} \
                                        --train-dataset ImageNet:split=TRAIN:root={base_dir}/{dataset_dir}/{dataset_name}:extra={base_dir}/{dataset_dir}/{dataset_name}/Extra \
                                        --val-dataset ImageNet:split=VAL:root={base_dir}/{dataset_dir}/{dataset_name}:extra={base_dir}/{dataset_dir}/{dataset_name}/Extra \
                                        --batch_size {batch_size} \
                                        --train_val_test_nums {train_val_test_num} \
                                        --image_res {img_res}'"
                    os.system(sys_cmd)
                    time.sleep(1)