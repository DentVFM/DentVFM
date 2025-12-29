import time
import os
from datetime import datetime
import random
import subprocess
import sys

# dataset configurations
kfold = 1
base_dir = "XXXX"
base_dataset_names=[
                    # "Dataset112_ToothFairy2",
                    # "Dataset113_NCseg",
                    # "Dataset114_Sts_miccia23",
                    # "Dataset115_Pal", 
                    # "Dataset116_HeadNeck_miccai15",    
                    # "Dataset117_Pulpy3D",
                    # "Dataset118_ToothFairy3"
                   ]
dataset_dirs = [
                # "Dataset112_ToothFairy2",
                # "Dataset113_NCseg",
                # "Dataset114_Sts_miccia23",
                # "Dataset115_Pal", 
                # "Dataset116_HeadNeck_miccai15",    
                # "Dataset117_Pulpy3D",
                # "Dataset118_ToothFairy3"
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
img_sizes = [
    "96",
    ] 
for model_id, model_dir in enumerate(model_dirs):
    ckp_dir = ckp_dirs[model_id]
    if ckp_dir[0] == "all" and len(ckp_dir) == 1:
        ckp_dir=os.listdir(f"{base_dir}/output_dir/XXXX/{model_dir}/eval")
        ckp_dir = [ckp for ckp in ckp_dir if not ckp.startswith(".ipynb_checkpoints")]
        ckp_dirs[model_id] = ckp_dir

# evaluation configurations
re_run = False
batch_size="2"

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
            img_size = img_sizes[model_id]
            
            for ckp_name in ckp_dir:
                print(f"[Seg Evaluation] Now is processing model_dir: {model_dir}/{ckp_name}")

                if os.path.exists(f"{base_dir}/output_dir/XXXX/{model_dir}/eval/{ckp_name}/seg_{dataset_name}/results.json") and re_run == False:
                    print("file exists! skip!")
                    continue
                else:
                    print("file not found, execute evaluation!")
                    sys_cmd = f"bash -c 'PYTHONPATH=. python dinov2/eval/segmentation3d.py \
                                        --config-file dinov2/configs/train/vitl3d_our.yaml \
                                        --pretrained-weights {base_dir}/output_dir/XXXX/{model_dir}/eval/{ckp_name}/teacher_checkpoint.pth \
                                        --output-dir {base_dir}/output_dir/XXXX/{model_dir}/eval/{ckp_name}/seg_{dataset_name} \
                                        --dataset-name {dataset_name} \
                                        --dataset-percent 100 \
                                        --base-data-dir {base_dir}/{dataset_dir} \
                                        --epochs 100 \
                                        --epoch-length 300 \
                                        --segmentation-head UNETR \
                                        --eval-iters 300 \
                                        --warmup-iters 3000 \
                                        --image-size {img_size} \
                                        --batch-size {batch_size} \
                                        --num-workers 10 \
                                        --learning-rate 1e-4 \
                                        --cache-dir {base_dir}/{dataset_dir}/{dataset_name}_{model_name}_cache \
                                        --resize-scale 1.0 \
                                        --model_name {model_name}'"
                    os.system(sys_cmd)
                    time.sleep(1)