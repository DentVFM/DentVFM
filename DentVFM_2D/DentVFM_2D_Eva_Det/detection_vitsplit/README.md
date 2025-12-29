# Applying ViT-Split to Object Detection

Our detection code is developed on top of [MMDetection v2.22.0](https://github.com/open-mmlab/mmdetection/tree/v2.22.0).

For details see [ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads](https://arxiv.org/pdf/2506.03433).

If you find our work helpful, please star this repo and cite our paper:

```
@article{li2025vit,
  title={ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads},
  author={Li, Yifan and Li, Xin and Li, Tianqin and He, Wenbin and Kong, Yu and Ren, Liu},
  journal={arXiv preprint arXiv:2506.03433},
  year={2025}
}
```

## Installation

Install [MMDetection v2.22.0](https://github.com/open-mmlab/mmdetection/tree/v2.22.0).

```
# recommended environment: torch1.9 + cuda11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0
pip install instaboostfast # for htc++
cd ops & sh make.sh # compile deformable attention
```

or install everything directly

```bash
. install.sh
```

## Data Preparation

Prepare COCO according to the guidelines in [MMDetection v2.22.0](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md).

## Pretraining Sources

| Name          | Type       | Year | Data         | Repo                                                                                                    | Paper                                                                                                                                                                           |
| ------------- | ---------- | ---- | ------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DINOv2        | Self-Supervised        | 2023 | LVD-142M | [repo](https://github.com/facebookresearch/dinov2)                                            | [paper](https://arxiv.org/abs/2208.06366)     |

## Training and Evaluation
Train and evaluate our ViT-Split refer to the following script
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 . dist_train.sh configs/mask_rcnn/vitsplit/mask_rcnn_dinov2_vitsplit_small_fpn_1x_coco.py.py 8 --seed 2023 
```

Or using the bash script directly:

```bash
# ViTSplit DINOv2-S
. scripts/vitsplit/train_dinov2-small_maskrcnn.sh
# ViTSplit DINOv2-B
. scripts/vitsplit/train_dinov2-base_maskrcnn.sh
# ViTSplit DINOv2-L
. scripts/vitsplit/train_dinov2-large_maskrcnn.sh
```

## Results

The results are expected to be consistent with these, though minor variations may occur across different machines.

| Method            | #Param | APᵇ | APᵇ₅₀ | APᵇ₇₅ | APᵐ | APᵐ₅₀ | APᵐ₇₅ |
|-------------------|--------|-----|-------|-------|-----|-------|-------|
| ViT-Split-S| 45M   | 48.5 | 70.5 | 53.3 | 42.8 | 67.2 | 45.6 |
| ViT-Split-B| 45M   | 51.8 | 73.6 | 57.1 | 45.4 | 70.3 | 48.6 |
| ViT-Split-L| 45M   | 53.0 | 75.1 | 58.1 | 46.6 | 71.9 | 50.4 |

## Acknowledgement

- This work is built upon the [ViT-Adapter](https://github.com/czczup/ViT-Adapter), Thanks!
