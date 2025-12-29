# 检查模型ckp中的key和shape，并将其保存到txt文件中
import torch
from torchio.transforms import RandomAffine

def check_ckp_keys_and_shapes(ckp_path, output_txt):
    model = torch.load(ckp_path, map_location='cpu')
    print(model.keys())
    print(model['teacher'].keys())
    with open(output_txt, 'w') as f:
        for key, value in model['teacher'].items():
            f.write(f"{key}: {value.shape}\n")
    print(f"Keys and shapes saved to {output_txt}")


if __name__ == "__main__":
    
    ckp_path = "/home/jovyan/dataset/Model_Outputs/dinov2/Fm_3d_Test0008/eval/training_11999/teacher_checkpoint.pth"
    output_txt = "./ckp_keys_and_shapes.txt"
    check_ckp_keys_and_shapes(ckp_path, output_txt)