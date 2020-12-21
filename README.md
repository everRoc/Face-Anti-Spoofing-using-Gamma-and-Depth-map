# Face-Anti-Spoofing-using-Gamma-and-Depth-map

1.训练模型：FAS_train.sh
涉及train_model.py，Load_data_train.py，Load_data_test.py，Load_data_test1.py等

2.测试模型：FAS_test.sh
支持jpg和JPG格式的图片;
涉及model_test.py，Load_data_test4.py等

3.其他：
finetune_model.py用于微调模型，涉及Load_data_train1.py，Load_data_test.py，Load_data_test1.py等

train_model_depth.py用于训练由人脸深度图辅助监督的模型,其中人脸深度图可基于PRnet生成，涉及Load_data_train2.py，Load_data_test3.py，Load_data_test2.py等

人脸检测和裁剪可基于MTCNN实现
