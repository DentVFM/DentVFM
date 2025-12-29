kill -9 $(lsof -t /dev/nvidia*)
sleep 1s
. dist_train.sh configs/mask_rcnn/vitsplit/mask_rcnn_dinov2_vitsplit_large_fpn_1x_coco.py 8 --seed 2023
