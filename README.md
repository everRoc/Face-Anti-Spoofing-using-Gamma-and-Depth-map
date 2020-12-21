# Face Anti-Spoofing using Gamma and Depth-map

---

## Introduction

The widespread deployment of face recognition-based biometric systems has made face Presentation Attack Detection(PAD), in other words, face anti-spoofing, an increasingly critical issue. Face anti-spoofing is usually combined with face recognition technology to form a complete system and deploy it to the real scene.


We use the modified [CDCN++](https://arxiv.org/pdf/2003.04092v1.pdf) network for face anti-spoofing. In the case that the generalization performance of the model trained directly with the complete input image is poor, we try to extract the face from the original training image by MTCNN for training the model (giving full play to the advantages of the model), and use Gamma transformation in the data enhancement stage to enhance the robustness of the model to the ambient light. As a result, the accuracy of model in OULU test set is 99.5%, and the performance in B_test is improved to 76.3%. At last, we introduce the face depth map as a auxiliary supervision, and the model keep the high accuracy rate (97.03%) of Oulu test set while the accuracy in the B_test is 89.18%.

---

### 训练模型（无深度图）：FAS_train.sh
涉及train_model.py，Load_data_train.py，Load_data_test.py，Load_data_test1.py等

### 测试模型：FAS_test.sh
支持jpg和JPG格式的图片;
涉及model_test.py，Load_data_test4.py等

### 其他：

* finetune_model.py用于微调模型，涉及Load_data_train1.py，Load_data_test.py，Load_data_test1.py等

* train_model_depth.py用于训练由人脸深度图辅助监督的模型,涉及Load_data_train2.py，Load_data_test3.py，Load_data_test2.py等,其中人脸深度图可基于[PRNet](https://github.com/spicy-dog/Face-Depth-map-Generation-using-PRNet)生成

* 人脸检测和裁剪可基于[MTCNN](https://github.com/spicy-dog/Crop-face-using-MTCNN)实现
