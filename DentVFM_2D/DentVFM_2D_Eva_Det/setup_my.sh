conda create --name cac_det_v1 python=3.9.7 -y
conda activate cac_det_v1

pip install notebook
python -m ipykernel install --user --name=cac_det_v1

# pip install "numpy<2" torch==2.0.0 torchvision==0.15.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install -U torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install -U segmentation-models

cd mmcv-1.5.0
TORCH_CUDA_ARCH_LIST="8.9" MMCV_WITH_OPS=1 FORCE_CUDA=1 python setup.py install
# 编辑cpp_extension.py，添加8.9支持

cd mmdetection
pip install --no-build-isolation -e .

cd mmsegmentation
pip install --no-build-isolation -e .

pip install scikit-learn scikit-multilearn tqdm matplotlib

pip install opencv-python timm==0.4.12 yapf==0.40.1

conda install -c conda-forge ipywidgets
conda install -c conda-forge notebook jupyterlab

cd detection_vitadapter/ops
bash make.sh

pip install -U openmim
mim install mmengine












conda create --name cac_det python=3.9.7 -y
conda activate cac_det

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install --no-build-isolation -v -e .

git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install --no-build-isolation -v -e .

pip install -U openmim
mim install mmengine
mim install "mmcv<2.2.0,>=2.0.0"

pip install scikit-learn scikit-multilearn tqdm matplotlib

pip install opencv-python timm==0.4.12 yapf==0.40.1