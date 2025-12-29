import argparse

import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('filename', nargs='?', type=str, default=None)

args = parser.parse_args()

model = torch.load(args.filename, map_location=torch.device('cpu'))
ori_ckp = model['teacher']
# 提取开头为 'backbone.' 的权重并移除前缀
tar_ckp = {}
block_chunck = 4

for key_n in ori_ckp.keys():
    if 'backbone' in key_n:
        # if 'gamma' in key_n:
        #     ori_key_n = key_n.replace('ls1.gamma','gamma1') if 'ls1' in key_n else key_n.replace('ls2.gamma','gamma2')
        # else:
        #     ori_key_n = key_n
        ori_key_n = key_n
        # 去掉block chunck
        for chunck_id in range(block_chunck):
            if f'blocks.{chunck_id}' in ori_key_n:
                print(f'chunck_id: {chunck_id}, ori_key_n: {ori_key_n}')
                ori_key_n = ori_key_n.replace(f'blocks.{chunck_id}', 'blocks')
                break

        key_new = ori_key_n.replace('backbone.','')
        tar_ckp[key_new] = ori_ckp[key_n]
        
# torch.save(tar_ckp, args.filename.replace('.pth', '_backbone.pth'))
torch.save(tar_ckp, args.filename.replace('.pth', '_backbone_ori.pth'))
