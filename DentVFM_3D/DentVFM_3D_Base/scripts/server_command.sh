## 使用ImageNet-1K进行训练测试
# 测试环境
baizectl job submit --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan8 --max-retries 1 -- python test_environment.py

# 生成extra文件
baizectl job submit --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan3 --max-retries 1 -- python prepare_imagenet_extra.py
baizectl job submit --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan8 --max-retries 1 -- python prepare_imagenet_extra.py

# 运行train 单个node
baizectl job submit --shm-size 10240 --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan6 --max-retries 1 -- bash ./scripts/imagenet_test_train.sh

# 运行train 多个node
baizectl job submit --shm-size 10240 --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan6 --max-retries 1 --workers 4 -- bash ./scripts/imagenet_test_train.sh

# 运行eval 单个node
baizectl job submit --shm-size 10240 --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan7 --max-retries 1 -- bash ./scripts/imagenet_test_eval.sh


## 使用9h_fm_data_2d进行训练测试
# 预处理origin数据，生成9h_fm_data_2d
baizectl job submit --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan8 --max-retries 1 -- python prepare_origin_data.py

# 预处理origin数据，生成9h_fm_data_2d多进程
baizectl job submit --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan8 --max-retries 1 -- python prepare_origin_data_mp.py

# 生成extra文件
baizectl job submit --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan8 --max-retries 1 -- python prepare_9h_fm_data_2d_extra.py

# 运行train，多个node
baizectl job submit --shm-size 10240 --image 10.1.17.1/m.daocloud.io/nvcr.io/nvidia/pytorch:24.02-py3-jiuyuan8 --max-retries 1 --workers 4 -- bash ./scripts/fm_2d_test_train.sh