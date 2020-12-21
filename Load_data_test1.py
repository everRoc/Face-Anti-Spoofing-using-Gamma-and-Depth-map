from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

from PIL import Image
from torch_py.FaceRec import Recognition

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, binary_mask, spoofing_label, fileName= sample['image_x'], sample['binary_mask'],sample['spoofing_label'],sample["fileName"]
        
        new_image_x = (image_x - 127.5)/128     # [-1,1]

        return {'image_x': new_image_x, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label,"fileName":fileName}

class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, binary_mask, spoofing_label, fileName= sample['image_x'], sample['binary_mask'],sample['spoofing_label'],sample["fileName"]
        
        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
        
        binary_mask = np.array(binary_mask)

                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'binary_mask': torch.from_numpy(binary_mask.astype(np.float)).float(), 'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.float)).float(),"fileName":fileName}



class Spoofing_test1(Dataset):

    def __init__(self, root_dir,  transform=None):

        # self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        
        self.root_dir = root_dir
        self.transform = transform
        #self.recognize = Recognition()

        '''
        if root_dir.split("oulu")[1] == "/devset/":
            
            self.root_depth = root_dir.split("oulu")[0] + "oulu/dev_depth1/"

        elif root_dir.split("oulu")[1] == "/trainset/":
            
            self.root_depth = root_dir.split("oulu")[0] + "oulu/train_depth1/"
        #print(self.root_depth)
        '''

    def __len__(self):
        dirList = os.listdir(self.root_dir)
        return len(dirList)

    
    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        # videoname = str(self.landmarks_frame.iloc[idx, 0])
        # image_path = os.path.join(self.root_dir, videoname)

        image_path = self.root_dir
        dirList = os.listdir(image_path)
        image_path = image_path + dirList[idx]
        #depth_path = self.root_depth + dirList[idx]
        #print(depth_path)

        # print(dirList[idx])
        # print(dirList[idx].split("_")[-1])
        # print(dirList[idx])
        # temp = dirList[idx].split(".jpg")[0]
        # print (temp)
        # label = temp.split()

        #label = int((dirList[idx].split(".")[0]).split("_")[-2])
        #'''
        if (dirList[idx].split(".")[0]).split("(")[0] == 'zheng ':
            label = 1
        else:
            label = 0
        #'''

        #print(dirList[idx],label)
        # imList = os.listdir(image_path)
        #image_path = image_path +"/" +dirList[idx]
        # print(image_path)
        image_x, binary_mask = self.get_single_image_x(image_path)

        
        if label == 1:
            spoofing_label = 1            # real
            
            #if not (cv2.imread(depth_path, 0) is None):
                #image_depth = cv2.imread(depth_path, 0)
                #image_depth = cv2.resize(image_depth, (32, 32))
                #binary_mask = image_depth/255.0
            '''
            print(binary_mask)
            plt.imshow(image_x)
            plt.show()
            result = torch.sum(torch.from_numpy(binary_mask.astype(np.float)).float())/(32*32)
            print(result)
            if result > 0.5:
                print("yes")
            '''
        else:
            spoofing_label = 0            # fake
            binary_mask = np.zeros((32, 32))


        #frequency_label = self.landmarks_frame.iloc[idx, 2:2+50].values

        sample = {'image_x': image_x, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label,"fileName":dirList[idx]}

        if self.transform:
            sample = self.transform(sample)
        return sample

#     def get_single_image_x(self, image_path):


#         image_x = np.zeros((256, 256, 3))
#         binary_mask = np.zeros((32, 32))


#         image_x_temp = cv2.imread(image_path)
#         image1 = image_x_temp
#         image_x_temp_gray = cv2.imread(image_path, 0)
#         doGamma = True
#         if doGamma:
#             image_x_temp = gammaTrans(image_x_temp)

#         '''
# #############    crop face    ######################
#         image_x_temp = Image.fromarray(cv2.cvtColor(image_x_temp, cv2.COLOR_BGR2RGB))

#         recognize = Recognition()

#         #draw = recognize.face_recognize(img)
#         #plot_image(draw)

#         if not recognize.crop_faces(image_x_temp):
#             image_x_temp = image1

#         else:
#             image_x_temp = recognize.crop_faces(image_x_temp)[0]

#         image_x_temp = cv2.cvtColor(np.asarray(image_x_temp), cv2.COLOR_RGB2BGR)

#         '''
#         image_x = cv2.resize(image_x_temp, (256, 256))
#         #image_x = image_x_temp

#         # plt.imshow(image_x)
#         # plt.show()
#         image_x_temp_gray = cv2.resize(image_x_temp_gray, (32, 32))
#         image_x_aug = seq.augment_image(image_x)


#         for i in range(32):
#             for j in range(32):
#                 if image_x_temp_gray[i,j]>0:
#                     binary_mask[i,j]=1
#                 else:
#                     binary_mask[i,j]=0




#         return image_x_aug, binary_mask

    def get_single_image_x(self, image_path):
        image_x = np.zeros((256, 256, 3))
        binary_mask = np.zeros((32, 32))
        image_x_temp = cv2.imread(image_path)
        image_x_temp_gray = cv2.imread(image_path, 0)
        image_x = cv2.resize(image_x_temp, (256, 256))
        image_x_temp_gray = cv2.resize(image_x_temp_gray, (32, 32))
        for i in range(32):
            for j in range(32):
                if image_x_temp_gray[i,j]>0:
                    binary_mask[i,j]=1
                else:
                    binary_mask[i,j]=0
        return image_x, binary_mask




'''
#rootDir = "/home/chang/dataset/oulu/train_face1/"
rootDir = "/home/chang/dataset/B_face1/"
d = Spoofing_train(rootDir,transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))

for i, sample_batched in enumerate(d):
    print(i)
    image_x = sample_batched['image_x']
    im = image_x.permute([1,2,0]).numpy()
    #plt.imshow(im)
    #plt.show()
    if i>110:
        break
'''
