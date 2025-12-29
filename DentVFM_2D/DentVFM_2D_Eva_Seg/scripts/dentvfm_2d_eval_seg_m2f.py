
import time
import os
from datetime import datetime


# dataset configurations
kfold = 5
base_dir = "XXXX"
re_run = False
mmseg_dir="/DentVFM/DentVFM_2D/DentVFM_2D_Eva_Seg"
model_dir="output_dir/XXXX"
dataset_type_name_list = [
    # {"dataset_type": "miccai23sts2d",
    #  "data_name":"miccai23sts2d_segmentation_v1"},
    # {"dataset_type": "dc100",
    #  "data_name":"DC1000_segmentation_v1"},
    # {"dataset_type": "duallabelpan",
    #  "data_name":"A_dual_labeled_dataset_segmentation_v1"}
]
model_ckp_config_list = [
    {
        "model_name": 'XXXX',
        "ckp": ['XXXX'],
        'config_file': 'ours_vitb14_config_hr518.py'
    }
]
for model_ckp_config in model_ckp_config_list:
    ckp_dir = model_ckp_config["ckp"]
    model_name = model_ckp_config["model_name"]
    if ckp_dir[0] == "all" and len(ckp_dir) == 1:
        ckp_dir=os.listdir(f"{base_dir}/output_dir/XXXX/{model_name}/eval")
        ckp_dir = [ckp for ckp in ckp_dir if not ckp.startswith(".ipynb_checkpoints")]
        model_ckp_config["ckp"] = ckp_dir

# info
print("# of total tasks:",len(dataset_type_name_list), "details:", dataset_type_name_list)
print("start evaluate [Seg]:")

for dataset_type_name in dataset_type_name_list:
    dataset_type=dataset_type_name['dataset_type']
    data_name=dataset_type_name['data_name']

    print("current dataset:", dataset_type, data_name)

    for fold_id in range(kfold):
        data_root = f"{data_name}_fold{fold_id+1}"
        print("current fold: ", data_root)

        for model_ckp_config in model_ckp_config_list:
            model_name=model_ckp_config["model_name"]
            ckp_names=model_ckp_config["ckp"]
            config_file=model_ckp_config["config_file"]

            for ckp in ckp_names:
                print("[M2F Seg Evaluation] Now is processing model_dir: ", model_name, ckp)

                if "Fm_2d_" in model_name:
                    backbone_ckp = f"{model_dir}/{model_name}/eval/{ckp}/teacher_checkpoint_segm2f_cs224.pth"
                else:
                    backbone_ckp = "keep_origin"

                sys_cmd = f"cd {mmseg_dir};\
                            bash -c 'python train.py \
                            --config ./configs/{dataset_type}/{config_file} \
                            --data_root {data_root} \
                            --pretrained {backbone_ckp} \
                            --work-dir {model_dir}/{model_name}/eval/{ckp}/segm2f_{data_root}'"
                
                os.system(sys_cmd)
                time.sleep(1)