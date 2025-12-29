from dinov2.data.datasets import ImageNet

# 9h_fm_data_2d_v1 v2 v3 v4 v5 v6
for split in ImageNet.Split:
    dataset = ImageNet(split=split, 
                       root="../../../Origin_Datasets/9h_fm_data_2d_v6/Data", 
                       extra="../../../Origin_Datasets/9h_fm_data_2d_v6/Extra")
    dataset.dump_extra()


# 下游任务: 模态识别 modality_recognition_2d_xray_only
# for split in ImageNet.Split:
#     dataset = ImageNet(split=split, 
#                        root="/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/modality_recognition/modality_recognition_2d_xray_only_v5", 
#                        extra="/home/jovyan/dataset/Origin_Datasets/9h_fm_downstream_tasks/modality_recognition/modality_recognition_2d_xray_only_v5/extra")
#     dataset.dump_extra()