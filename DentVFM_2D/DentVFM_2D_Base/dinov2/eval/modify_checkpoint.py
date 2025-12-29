"""
mmseg要加载模型时需要将ckp的key进行调整

block chunck统一去除

mmseg+bnhead:
1. teacher去掉
2. blockchunk要添加参数控制
3. backbone去掉
4. dino_head忽略
5. 结果保存为 后缀 _segbnhead_csxxx.pth  cs为crop size控制pos embedding
"""

import torch
import os

if __name__ == '__main__':

    # model_base = '/mnt/ugreenf8c5_samba_share/huangxinrui/Model_Outputs/dinov2'
    model_base = '/home/jovyan/dataset/Model_Outputs/dinov2'
    # model_names = [
    #     "Fm_2d_Test0008-1",
    #     "Fm_2d_Test0008-2",
    #     "Fm_2d_Test0008-3",
    #     "Fm_2d_Test0010-2",
    #     "Fm_2d_Test0010-3",
    #     "Fm_2d_Test0011-1",
    #     "Fm_2d_Test0011-4"
    # ]
    model_names = [
        "Fm_2d_Test0011-2"
    ]

    # 针对mmseg+bhead修改ckp
    for model_name in model_names:
        model_eval_path = os.path.join(model_base, model_name, "eval")

        print("model_name", model_name)
        
        for ckp_name in os.listdir(model_eval_path):
            # 避免ipynb_checkpoints的影响
            if 'ipynb_checkpoints' in ckp_name or ckp_name.startswith('.'):
                continue
            print("ckp_name", ckp_name)

            ori_ckp_file = os.path.join(model_eval_path, ckp_name, "teacher_checkpoint.pth")
            out_ckp_file = os.path.join(model_eval_path, ckp_name, "teacher_checkpoint_segbnhead_cs224.pth")

            checkpoint = torch.load(ori_ckp_file, map_location="cpu")
            state_dict = checkpoint['teacher']
            backbone_state_dict = {}
            # 如果模型分块block chunck，需要去除chunck
            flag = False
            for key, value in state_dict.items():
                if 'blocks.0.0.' in key:
                    flag = True
                    break
            for key, value in state_dict.items():
                if 'dino_head' in key or 'ibot_head' in key:
                    continue
                else:
                    if flag:
                        # 去掉block chunck
                        for chunck_id in range(4):
                            if f'blocks.{chunck_id}' in key:
                                key = key.replace(f'blocks.{chunck_id}', 'blocks')
                                break
                    backbone_state_dict[key] = value
            backbone_state_dict = {k.replace("module.", ""): v for k, v in backbone_state_dict.items()}
            backbone_state_dict = {k.replace("backbone.", ""): v for k, v in backbone_state_dict.items()}
            
            torch.save(backbone_state_dict, out_ckp_file)