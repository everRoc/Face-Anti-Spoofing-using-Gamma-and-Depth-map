from __future__ import print_function, division
import cv2
import torch
import matplotlib as mpl
# mpl.use('TkAgg')
import argparse, os
import pandas as pd
from PIL import Image
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.CDCNs import Conv2d_cd, CDCNpp, CDCN, CDCNpp1
from Load_data_test4 import Spoofing_test1, Normaliztion, ToTensor

# from utils import AvgrageMeter, accuracy, performances
# import multiprocessing as mp
from torch_py.FaceRec import Recognition

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')
    parser.add_argument('--path', type=str, default="E:/dataset/testoulu/B_test", help='test_data_path')
    parser.add_argument('--predictionFile', type=str, default="prediction.csv", help='prediction file path')
    # parser.add_argument('--path', type=str, help='test_data_path')
    # parser.add_argument('--predictionFile', type=str, help='prediction file path')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

################# crop face#############################
    pnet_path = "./torch_py/MTCNN/weights/pnet.npy"
    rnet_path = "./torch_py/MTCNN/weights/rnet.npy"
    onet_path = "./torch_py/MTCNN/weights/onet.npy"

    image_folder = args.path
    save_folder = args.path + "/crop_face_data"
    predictionFile = args.predictionFile
    fPredictionFile = open(predictionFile, "w") 
    fPredictionFile.write("pic_id,pred\n")

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    first_dir = os.listdir(image_folder)
    num_img = 0
    # num_failed = 0   # 提取人脸失败的图片数量
    # num_right = 0    # 非活体的失败图片数目
    for img in first_dir:

        if img.split(".")[-1] == 'jpg' or img.split(".")[-1] == 'JPG':
            path_image = image_folder + '/' + str(img)
            image = Image.open(path_image)

            num_img += 1
            print(num_img, str(img))

            recognize = Recognition()
            # draw = recognize.face_recognize(img)
            # plot_image(draw)
            if not recognize.crop_faces(image):
                newLine = str(img) + ",%f\n" % (0)
                fPredictionFile.write(newLine)

                # face_img = image
                print("Failed!!")
                # num_failed += 1
                ################  直接将未检测到人脸的图片判为非活体
                #if (img.split(".")[0]).split("(")[0] != 'zheng ':
                    #num_right += 1
                continue
                # plot_image(image)
            else:
                face_img = recognize.crop_faces(image)[0]
                save_path = save_folder + '/' + str(img)
                face_img.save(save_path)

    # num_directly_N = num_right
    # print("num_directly_TN: %d" % num_directly_N)
    # print("num_failed: %d" % num_failed)

    test_face_path = save_folder + "/"

    model = CDCNpp(basic_conv=Conv2d_cd, theta=args.theta)
    # model = CDCN( basic_conv=Conv2d_cd, theta=args.theta)
    # model = CDCNpp1( basic_conv=Conv2d_cd, theta=args.theta)

    model.load_state_dict(torch.load('load_model.pkl'))##########################################
    model = model.cuda()

    # print(model)
    model.eval()

    # meanAcT = []
    # meanAcF = []
    with torch.no_grad():

        val_data = Spoofing_test1(test_face_path,
                                   transform=transforms.Compose([Normaliztion(), ToTensor()]))

        test_ba = 1
        dataloader_val = DataLoader(val_data, batch_size=test_ba, shuffle=False, num_workers=4)

        # num_true, num_false = 0,0

        # threshold = 0.36
        # threshold = 0.5
        for i, sample_batched in enumerate(dataloader_val):
            #             # get the inputs
            inputs, binary_mask, spoof_label,fileName = sample_batched['image_x'].cuda(), sample_batched['binary_mask'].cuda(), \
                                               sample_batched['spoofing_label'].cuda(),sample_batched['fileName']

            # optimizer.zero_grad()
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input = model(inputs)
            # map_x shape: batch,N,N
            for ba in range(test_ba):
                #pre_label = 0
                map_score = torch.sum(map_x[ba]) / (32 * 32)

                newLine = str(fileName) + ",%f\n" % (map_score.item())
                fPredictionFile.write(newLine)
                '''
                if map_score >= threshold:
                    pre_label = 1

                if pre_label == spoof_label:
                    num_right += 1
                if spoof_label != 1:
                    meanAcF.append(1 - map_score.item())
                else:
                    meanAcT.append(map_score.item())
                '''
    '''
    # print("TP",np.mean(meanAcT))
    # print("TN",np.mean(meanAcF))
    meanAcT = np.array(meanAcT)
    meanAcF = np.array(meanAcF)
    TP = len(meanAcT[meanAcT > threshold])
    TN = len(meanAcF[meanAcF > 1 - threshold]) + num_directly_N
    acc = (TP + TN) / (len(meanAcF) + len(meanAcT) + num_failed)
    accuracy = num_right / (len(dataloader_val) + num_failed)
    print("ACC", acc, ":T ", TP, ":", len(meanAcT) + num_failed - num_directly_N, " F ", TN, ":", len(meanAcF) + num_directly_N)
    print("ACCURACY:", accuracy)
    '''
    fPredictionFile.close()

    # print(args.log + '/' + args.log)

#   所要求输出为csv文件，每行为图片id及对应置信度，因此裁图片阶段可能需要存下失败图片的id

