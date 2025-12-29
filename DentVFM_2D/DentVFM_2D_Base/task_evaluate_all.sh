# 评测所有数据集的bash

## 手术类型预测
bash ./scripts/task_cls_9h_operation_before_type_classification_server.sh >./task_evaluation_logs/evaluation_bash_log/9h_single_double_before_xray_v1_log.txt 2>&1 &

bash ./scripts/task_cls_9h_operation_after_type_classification_server.sh >./task_evaluation_logs/evaluation_bash_log/9h_single_double_after_xray_v1_log.txt 2>&1 &


## 年龄预测
bash ./scripts/task_cls_9h_age_lat_classification_server.sh >./task_evaluation_logs/evaluation_bash_log/9h_age_classification_lateral_v1_log.txt 2>&1 &
## (重新处理过数据，应该全部重跑)
bash ./scripts/task_cls_9h_age_pan_classification_server.sh >./task_evaluation_logs/evaluation_bash_log/9h_age_classification_panoramic_v1_log.txt 2>&1 &


## 囊肿类型分类
bash ./scripts/task_cls_9h_cyst_type_classification_server.sh >./task_evaluation_logs/evaluation_bash_log/9h_cyst_tumor_cls_4type_pan_v1_log.txt 2>&1 &


## 骨折类型诊断分类
bash ./scripts/task_cls_9h_fracture_type_classification_server.sh >./task_evaluation_logs/evaluation_bash_log/9h_fracture_type_classification_xray_pan_v1_log.txt 2>&1 &


## 牙周炎分级
bash ./scripts/task_cls_9h_periodontal_grade_classification_server.sh >./task_evaluation_logs/evaluation_bash_log/9h_periodontal_grade_classification_v1_log.txt 2>&1 &


## 畸形类型分类(重新处理过数据，应该全部重跑)
bash ./scripts/task_cls_anatomical_type_classification_server.sh >./task_evaluation_logs/evaluation_bash_log/anatomical_type_classification_ANB_v2_log.txt 2>&1 &


## 根尖手术
bash ./scripts/task_cls_apicoectomy_surgery_panoramic_server.sh >./task_evaluation_logs/evaluation_bash_log/apicoectomy_surgery_panoramic_cls_v2_log.txt 2>&1 &


## patch牙齿异常分类
bash ./scripts/task_cls_patch_teeth_diagnosis_dentex_classification_server.sh >./task_evaluation_logs/evaluation_bash_log/teeth_patch_diagnosis_cls_miccai23_DENTEX_task3_v1_log.txt 2>&1 &

bash ./scripts/task_cls_patch_teeth_diagnosis_drad_classification_server.sh >./task_evaluation_logs/evaluation_bash_log/teeth_patch_diagnosis_cls_DRAD_Kaggle_v1_log.txt 2>&1 &