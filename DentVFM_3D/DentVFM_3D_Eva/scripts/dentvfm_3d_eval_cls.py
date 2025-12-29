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
                    # "9h_fracture_classification_ct_cbct_v1",
                    # "9h_orsplit_after_ctcbct_v1",
                    # "9h_orsplit_before_ctcbct_v1",
                    # "9h_single_double_after_ctcbct_v1", 
                    # "9h_single_double_before_ctcbct_v1",    
                    # "9h_tmj_abnormal_classification_v1"
                   ]
dataset_dirs = [
                # "9h_fracture_classification_ct_cbct",
                # "9h_orsplit_after_ctcbct",
                # "9h_orsplit_before_ctcbct",
                # "9h_single_double_after_ctcbct", 
                # "9h_single_double_before_ctcbct",    
                # "9h_tmj_abnormal_classification"
               ]

# model configurations
model_dirs = [
            "XXXX"
           ]
ckp_dirs = [
          ["XXXX"]
         ]
model_names = [
            "XXXX"
        ]
config_names = [
            "vitb3d_our.yaml"
        ]
for model_id, model_dir in enumerate(model_dirs):
    ckp_dir = ckp_dirs[model_id]
    if ckp_dir[0] == "all" and len(ckp_dir) == 1:
        ckp_dir=os.listdir(f"{base_dir}/output_dir/XXXX/{model_dir}/eval")
        ckp_dir = [ckp for ckp in ckp_dir if not ckp.startswith(".ipynb_checkpoints")]
        ckp_dirs[model_id] = ckp_dir

# evaluation configurations
re_run = False
batch_size="6"

# info
print("# of total tasks:",len(base_dataset_names), "details:", base_dataset_names)


for idx,base_dataset_name in enumerate(base_dataset_names):
    dataset_dir = dataset_dirs[idx]

    dataset_names=[]
    for fold_id in range(kfold):
        dataset_names.append(f"{base_dataset_name}_fold{fold_id+1}")

    print(f"current dataset: {base_dataset_name}, all folds: {dataset_names}")
    
    for dataset_name in dataset_names:
        print(f"current fold: {dataset_name}")
        
        for model_id, model_dir in enumerate(model_dirs):
            ckp_dir = ckp_dirs[model_id]
            model_name = model_names[model_id]
            config_name = config_names[model_id]
            
            for ckp_name in ckp_dir:
                print(f"[Linear Evaluation] Now is processing model_dir: {model_dir}/{ckp_name}")

                if os.path.exists(f"{base_dir}/output_dir/XXXX/{model_dir}/eval/{ckp_name}/linear_{dataset_name}/results_eval_linear.json") and re_run == False:
                    print("file exists! skip!")
                    continue
                else:
                    print("file not found, execute evaluation!")
                    sys_cmd = f"bash -c 'PYTHONPATH=. python dinov2/eval/linear3d.py \
                                        --config-file dinov2/configs/train/{config_name} \
                                        --pretrained-weights {base_dir}/output_dir/XXXX/{model_dir}/eval/{ckp_name}/teacher_checkpoint.pth \
                                        --output-dir {base_dir}/output_dir/XXXX/{model_dir}/eval/{ckp_name}/linear_{dataset_name} \
                                        --dataset-name {dataset_name} \
                                        --dataset-percent 100 \
                                        --base-data-dir {base_dir}/{dataset_dir} \
                                        --epochs 100 \
                                        --epoch-length 125 \
                                        --save-checkpoint-frequency 20 \
                                        --eval-period-iterations 20 \
                                        --image-size 128 \
                                        --batch-size {batch_size} \
                                        --num-workers 10 \
                                        --dataset-seed 123 \
                                        --cache-dir {base_dir}/{dataset_dir}/{dataset_name}_{model_name}_cache \
                                        --model-name {model_name}'"
                    os.system(sys_cmd)
                    time.sleep(1)