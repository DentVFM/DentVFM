# Towards Dental Generalist Intelligence: Versatile Vision Foundation Models for Oral and Maxillofacial Radiology


<p align="center">
    <img src="./docs/static/images/dentvfms_logo.png" width="400"/>
<p>

<p align="center">
        📑 <a href="https://arxiv.org/abs/2510.14532">Paper</a>&nbsp&nbsp | &nbsp&nbsp🖥️ <a href="https://eliahuhorwitz.github.io/Academic-project-page-template/">Demo</a>&nbsp&nbsp | &nbsp&nbsp✍️ <a href="#citation">BibTex</a>&nbsp&nbsp
</p>


## Introduction
DentVFM — a family of pre-trained versatile vision foundation models for oral and maxillofacial radiology.

DentVFM is designed to produce task-agnostic representations for diverse dental applications, including diagnosis, treatment analysis, biomarker identification, landmark detection, and lesion&anatomy segmentation. 
DentVFM is pre-trained with self-supervised learning (SSL) on DentVista, one of the largest multimodal dental radiology imaging datasets. 
Additionally, a comprehensive dental benchmark, DentBench, is constructed to evaluate DentVFM in four dimensions: dental generalist intelligence (versatility), label efficiency, scalability, and cross-modality diagnosis. 
DentVFM provides a label-efficient, generalizable, and scalable foundation to advances intelligent oral healthcare.


#### Main contributions

* **Model Family**: A series of models pre-trained using SSL with large scale data, tailored to different data dimensionalities.

* **DentBench**: A novel, comprehensive, multi-center, multi-task and multi-disease benchmark that includes both public tasks and carefully curated tasks emphasizing treatment analysis and biomarker identification. 

* **Four Key Advantages**: Versatility, label efficiency, scalability, and cross-modality diagnosis.


#### Main framework

<p align="center">
    <img src="./docs/static/images/main_framework.png" width="70%"/>
<p>


#### Overall results
<p align="center">
    <img src="./docs/static/images/overall_results_v2.png" width="70%"/>
<p>


## News
* 2025.12.22: We are releasing the training and evaluation code. 

## Family of models
Multiple versions of DentVFM are provided. 

Note: After peer review, the weights will be made publicly available; 
during this period, they will be provided selectively upon request (<a href="./docs/static/pdfs/application_form.pdf">Protocol</a>).

<div align="center">
<table style="border-collapse:collapse; text-align:center;">
  <thead>
    <tr>
      <th style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">model</th>
      <th style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">version</th>
      <th style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;"># of params</th>
      <th style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;"><b>DentVFM-2D</b></td>
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">ViT-Base</td>
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">86M</td>
      <!-- <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;"><a href="preparing">Access</a></td> -->
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">Access</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">ViT-Large</td>
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">300M</td>
      <!-- <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;"><a href="preparing">Access</a></td> -->
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">Access</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">ViT-Giant</td>
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">1100M</td>
      <!-- <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;"><a href="preparing">Access</a></td> -->
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">Access</td>
    </tr>
    <tr>
      <td rowspan="3" style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;"><b>DentVFM-3D</b></td>
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">ViT-Base</td>
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">86M</td>
      <!-- <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;"><a href="preparing">Access</a></td> -->
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">Access</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">ViT-Large</td>
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">300M</td>
      <!-- <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;"><a href="preparing">Access</a></td> -->
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">Access</td>
    </tr>
    <tr>
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">ViT-Giant</td>
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">1100M</td>
      <!-- <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;"><a href="preparing">Access</a></td> -->
      <td style="padding:6px 10px; border:1px solid #ddd; vertical-align:middle;">Access</td>
    </tr>
  </tbody>
</table>
</div>

## DentBench
DentBench consists of publicly available dental datasets and carefully curated datasets. 
Details can be found in here.
Access links for the public datasets are provided.
Please use them in compliance with the agreements specified by the owners.

Note: The curated datasets will be made publicly available after the peer-review process; 
during this period, they will be provided selectively upon request (<a href="./docs/static/pdfs/application_form.pdf">Protocol</a>).


## How to pretrain

### Installation
Create a virtual environment and install the dependencies required for pre-training the 2D and 3D models. 
Please refer to `./requirements/requirements_base.txt` for versions of packages.

### Data preparation

#### 2D imaging dataset
Please construct the 2d pre-training dataset according to the ImageNet-1k format.
The root directory of the dataset should hold the following contents:
```
|-/2d_dataset_root

|--/Data
|---/train
|----/class1
|-----/class1_10202100000000000.JPEG
|-----/[..]
|----/classX
|---/val
|----/class1
|-----/ILSVRC2012_val_00000801.JPEG
|-----/[..]
|----/classX
|---/test
|----/ILSVRC2012_test_00000001.JPEG
|----/[..]
|---/labels.txt

|--/Extra
|---/class-ids-TRAIN.npy
|---/class-ids-VAL.npy
|---/class-names-TRAIN.npy
|---/class-names-VAL.npy
|---/entries-TEST.npy
|---/entries-TRAIN.npy
|---/entries-VAL.npy
```
Run the following command to generate `Extra` directory:
```shell
cd /DentVFM/DentVFM_2D/DentVFM_2D_Base
python prepare_dataset_extra.py --dataset_name dentvista_2d_pretrain_dataset --train_num XXX --val_num XXX --test_num XXX
```

#### 3D imaging dataset
Please construct the 3d pre-training dataset directory following contents:
```
|-/3d_dataset_root
|--/train.json
|--/val.json
|--/test.json
```
Each json file is a list of dictionaries, where each dictionary maps to an image path like:
```json
[
    {"img":"/data_root/CBCT/CBCT_10202205939010000.mha"},
    {"img":"/data_root/TMJ/TMJ_10202203281010000.mha"},
    {"img":"/data_root/CT/CT_10202101950000000.mha"},
]
```


### Training

#### DentVFM-2D training
Run the following command to train DentVFM-2D.
```bash
cd /DentVFM/DentVFM_2D/DentVFM_2D_Base
# vitb14
bash ./scripts/dentvfm_2d_train_vitb14.sh
# vitl14
bash ./scripts/dentvfm_2d_train_vitl14.sh
# vitg14
bash ./scripts/dentvfm_2d_train_vitg14.sh
```


#### DentVFM-3D training
Run the following command to train DentVFM-3D.
```bash
cd /DentVFM/DentVFM_2D/DentVFM_3D_Base
# vitb14
bash ./scripts/dentvfm_3d_train_vitb14.sh
# vitl14
bash ./scripts/dentvfm_3d_train_vitl14.sh
# vitg14
bash ./scripts/dentvfm_3d_train_vitg14.sh
```


## How to evaluate

### DentVFM-2D evaluation

#### Classification

Here you can use the same virtual environment used in pretraining.
Please following the same format described in the section <a href="###2D imaging dataset">2D imaging dataset</a>

Run the following command to perform the evaluation.
```bash
cd /DentVFM/DentVFM_2D/DentVFM_2D_Base
# logistic regression
python ./scripts/dentvfm_2d_eval_cls_lr.py
# KNN
python ./scripts/dentvfm_2d_eval_cls_knn.py
# linear classification with data augmentation
python ./scripts/dentvfm_2d_eval_cls_linear.py
```

#### Segmentation via linear head

For segmentation evaluation, we utilize MMSegmentation. 
We recommend creating a new virtual environment and referring to `./requirements/requirements_seg_2d.txt`.
Please prepare the segmentation dataset following the <a href="https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/2_dataset_prepare.html">ADEChallengeData2016</a> format.

Run the following command to perform the evaluation.
```bash
cd /DentVFM/DentVFM_2D/DentVFM_2D_Base
# linear segmentation with BN
python ./scripts/dentvfm_2d_eval_seg_linear.py
```

#### Segmentation enhanced with parameter efficient fine-tuning

Here you can use the same virtual environment and dataset format used in the section <a href="###Segmentation via linear head">Segmentation via linear head</a>.

Run the following command to perform the evaluation.
```bash
cd /DentVFM/DentVFM_2D/DentVFM_2D_Eva_Seg
# vit-adapter enhanced mask2former
python ./scripts/dentvfm_2d_eval_seg_m2f.py
```

#### Landmark detection enhanced with parameter efficient fine-tuning
Here you can use the same virtual environment used in the section <a href="###Segmentation via linear head">Segmentation via linear head</a>.

Run the following command to perform the evaluation.
```bash
cd /DentVFM/DentVFM_2D/DentVFM_2D_Eva_Loc
# prepare dataset
python step1_dataset_split.py
# train and valid
python step2_train_and_valid_landmarkloc.py --fold XX --model_name XXX --save_model_dir /XXXX --model_config ./config/ours.py --image_width 1024 --image_height 1024
```


#### Detection enhanced with parameter efficient fine-tuning
Please create a new virtual environment and referring to `./requirements/requirements_det_2d.txt`.
Please organize the detection dataset following the COCO format.

Run the following command to perform the evaluation.
```bash
cd /DentVFM/DentVFM_2D/DentVFM_2D_Eva_Det
python train.py --fold X
```


### DentVFM-3D evaluation

#### Classification
Here you can use the same virtual environment and dataset format used in pretraining.

Run the following command to perform the evaluation.
```bash
cd /DentVFM/DentVFM_2D/DentVFM_3D_Eva
python dentvfm_3d_eval_cls.py
```

#### Segmentation via UNETR head

Here you can use the same virtual environment used in pretraining.
Please prepare the segmentation data according to the data format of MONAI.
Here, we present an example of a directory structure.
```
|-/3d_seg_dataset_root
|--/imagesTr
|---/XXXX_001_0000.mha
|---/[...]
|--/labelsTr
|---/XXXX_001.mha
|---/[...]
|--/datalist.json
```
`datalist.json` is a dictionary that contains lists representing the training, validation, and test splits.
```json
{
'training': [
  {
    'image':"/3d_seg_dataset_root/imagesTr/XXXX_128_0000.mha",
    'label':"/3d_seg_dataset_root/labelsTr/XXXX_128.mha"
  },
],
'validation':[
  {
    'image':"/3d_seg_dataset_root/imagesTr/XXXX_389_0000.mha",
    'label':"/3d_seg_dataset_root/labelsTr/XXXX_389.mha"
  },
],
'test':[
  {
    'image':"/3d_seg_dataset_root/imagesTr/XXXX_340_0000.mha",
    'label':"/3d_seg_dataset_root/labelsTr/XXXX_340.mha"
  },
]
}
```

Run the following command to perform the evaluation.
```bash
cd /DentVFM/DentVFM_2D/DentVFM_3D_Eva
python dentvfm_3d_eval_seg.py
```




## Acknowledgements
This project is built upon following excellent repositories.

<a href="https://github.com/facebookresearch/dinov2">dinov2</a>,
<a href="https://github.com/AICONSlab/3DINO">3DINO</a>,
<a href="https://github.com/czczup/ViT-Adapter">ViT-Adapter</a>,
<a href="https://github.com/JackYFL/ViT-Split">ViT-Split</a>,
<a href="https://github.com/open-mmlab/mmdetection">mmdetection</a>,
<a href="https://github.com/open-mmlab/mmsegmentation">mmsegmentation</a>,
<a href="https://github.com/MIC-DKFZ/nnUNet">nnUNet</a>,
<a href="https://github.com/tamasino52/UNETR">UNETR</a>


## License
All code and model weights are released under the CC BY-NC-ND 4.0 license.
Please visit this <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">link</a> for more details about the license.


## Citation
If you find our paper and code useful in your research, please consider giving a star 🌟 and citation ✍️.

```BibTeX
@article{huang2025towards,
  title={Towards Generalist Intelligence in Dentistry: Vision Foundation Models for Oral and Maxillofacial Radiology},
  author={Huang, Xinrui and Xiao, Fan and He, Dongming and Gao, Anqi and Li, Dandan and Zhang, Xiaofan and Zhang, Shaoting and Wang, Xudong},
  journal={arXiv preprint arXiv:2510.14532},
  year={2025}
}
```
<br>