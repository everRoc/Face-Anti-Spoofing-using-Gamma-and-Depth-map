# Face Anti-Spoofing using Gamma and Depth-map

近年来人脸识别在许多交互式人工智能系统中得到了广泛的应用。然而，人 脸识别技术易受打印照片攻击或电子屏幕攻击的影响，因此人脸活体检测技术通 常与人脸识别技术相结合，形成一个完整的系统并部署到真实场景中。


我们将修改后的 CDCN++网络模型用于活体检测任务。 在直接用完整的输入图像训练所得模型泛化性能较差的情况下，我们尝试从原训 练图片中通过 MTCNN 提取人脸用于训练模型（发挥模型优势），并在数据增强阶段进行 Gamma 变换（增强模型对环境光照的鲁棒性），所得模型在 OULU 测试集上保持较好的准确率（99.5%），在 B_test 上准 确率也得到了一定的提升，达到 76.3%；最后我们引入人脸深度图进行监督，在尽量保持 OULU 测试集高准确率 （97.03%）的同时，模型在 B_test 上的准确率达到了 89.18%。


1.训练模型（无深度图）：FAS_train.sh
涉及train_model.py，Load_data_train.py，Load_data_test.py，Load_data_test1.py等

2.测试模型：FAS_test.sh
支持jpg和JPG格式的图片;
涉及model_test.py，Load_data_test4.py等

3.其他：
finetune_model.py用于微调模型，涉及Load_data_train1.py，Load_data_test.py，Load_data_test1.py等

train_model_depth.py用于训练由人脸深度图辅助监督的模型,其中人脸深度图可基于PRnet生成，涉及Load_data_train2.py，Load_data_test3.py，Load_data_test2.py等

人脸检测和裁剪可基于MTCNN实现
