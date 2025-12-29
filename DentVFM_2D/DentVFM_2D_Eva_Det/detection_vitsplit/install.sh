conda create --name vitsplit python=3.8 -y
conda activate vitsplit
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.22
pip install mmdet==2.22.0 # for Mask2Former
pip install ipdb
pip install mmsegmentation==0.20.2
pip install scipy
pip install yapf==0.20.0
cd ops
. make.sh # compile deformable attention
cd ..
