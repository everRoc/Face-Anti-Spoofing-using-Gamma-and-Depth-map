from __future__ import print_function, division
import cv2
import torch
import matplotlib as mpl
# mpl.use('TkAgg')
import argparse,os
import pandas as pd

import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import multiprocessing as mp
 
from models.CDCNs import Conv2d_cd, CDCNpp, CDCN, CDCNpp1

from Load_data_train2 import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Load_data_test2 import Spoofing_val
from Load_data_test3 import Spoofing_test

import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

from utils import AvgrageMeter, accuracy, performances
import matplotlib.pyplot as plt



def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''
    

    kernel_filter_list =[
                        [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                        [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                        [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                        ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)
    
    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().cuda()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
    return contrast_depth


class Contrast_depth_loss(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss,self).__init__()
        return
    def forward(self, out, label): 

        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)
        
        
        criterion_MSE = nn.MSELoss().cuda()
    
        loss = criterion_MSE(contrast_out, contrast_label)
        #loss = torch.pow(contrast_out - contrast_label, 2)
        #loss = torch.mean(loss)
    
        return loss


# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log.txt', 'w')
    
    echo_batches = args.echo_batches

    print("Oulu-NPU, P1:\n ")

    log_file.write('Oulu-NPU, P1:\n ')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?
    finetune = args.finetune
    if finetune==True:
        print('finetune!\n')


    else:
        print('train from scratch!\n')
        log_file.write('train from scratch!\n')
        log_file.flush()
         

        #model = CDCNpp( basic_conv=Conv2d_cd, theta=0.7)
        model = CDCNpp( basic_conv=Conv2d_cd, theta=args.theta)
        #model = CDCN( basic_conv=Conv2d_cd, theta=args.theta)
        #model = CDCNpp1( basic_conv=Conv2d_cd, theta=args.theta)

        model = model.cuda()

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    print(model) 
    
    
    criterion_absolute_loss = nn.MSELoss().cuda()
    criterion_contrastive_loss = Contrast_depth_loss().cuda()


    ACER_save = 1.0
    
    MSELoss_list = []
    
    Contrast_depth_loss_list = []
    accuracy_list = []
    accuracy1_list = []
    total_loss_list = []
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        #meanLoss = []
        
        scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        
        loss_absolute = AvgrageMeter()
        loss_contra =  AvgrageMeter()
        loss_total = AvgrageMeter()

        model.train()
        
        # load random 16-frame clip data every epoch


        train_data = Spoofing_train("/home/chang/dataset/oulu/train_face1/", transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
        #/home/chang/dataset/oulu/train_face1/
        #train_data = Spoofing_train("../../../../trainset", transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)

        for i, sample_batched in enumerate(dataloader_train):
            # get the inputs
            inputs, binary_mask, spoof_label = sample_batched['image_x'].cuda(), sample_batched['binary_mask'].cuda(), sample_batched['spoofing_label'].cuda() 
            # inputs, binary_mask, spoof_label = sample_batched['image_x'], sample_batched['binary_mask'], sample_batched['spoofing_label']

            optimizer.zero_grad()

            
            # forward + backward + optimize
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs)
            
            #pdb.set_trace()
            #pdb.set_trace()
            absolute_loss = criterion_absolute_loss(map_x, binary_mask)
            

            contrastive_loss = criterion_contrastive_loss(map_x, binary_mask)
            

            loss =  absolute_loss + contrastive_loss
            
            #loss.update(loss.item(), n)
            #total_loss_list.append(loss.item())

            loss.backward()
            
            optimizer.step()
            
            n = inputs.size(0)
            loss_absolute.update(absolute_loss.data, n)
            loss_contra.update(contrastive_loss.data, n)
            loss_total.update(loss.data, n)
            #total_loss.append(loss.item())
            torch.cuda.empty_cache()


            if i % echo_batches == echo_batches-1:    # print every 50 mini-batches

                print('epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f' % (epoch + 1, i + 1, lr,  loss_absolute.avg, loss_contra.avg))
        
        #     #break
        # # whole epoch average
        print('epoch:%d, Train:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f, loss= %.4f \n' % (epoch + 1, loss_absolute.avg, loss_contra.avg,loss_total.avg))
        MSELoss_list.append(loss_absolute.avg)
        Contrast_depth_loss_list.append(loss_contra.avg)
        total_loss_list.append(loss_total.avg)

        log_file.write('epoch:%d, Train: Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f \n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        log_file.write('loss= %.4f \n' % loss_total.avg)
        log_file.flush()
           
        threshold = 0.2#########################
        # epoch_test = 1
        # if epoch>25 and epoch % 5 == 0:  
        if True:  
            model.eval()
            meanAcT = []
            meanAcF = []
            with torch.no_grad():

        #rootDir = "D:/dataset/oulu/oulu/trainset/"
                val_data = Spoofing_val("/home/chang/dataset/oulu/dev_face1/", transform=transforms.Compose([ToTensor(), Normaliztion()]))
                #/home/chang/dataset/oulu/dev_face1/
                #val_data = Spoofing_train("../../../../devset", transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
                # val_data = Spoofing_valtest(image_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                test_ba = 1
                dataloader_val = DataLoader(val_data, batch_size=test_ba, shuffle=False, num_workers=4)
                
                # map_score_list = []

                num = 0
                
                for i, sample_batched in enumerate(dataloader_val):
        #             # get the inputs
                    inputs, binary_mask, spoof_label = sample_batched['image_x'].cuda(), sample_batched['binary_mask'].cuda(), sample_batched['spoofing_label'].cuda() 
                    # inputs = sample_batched['image_x'].cuda()
                    # binary_mask = sample_batched['binary_mask'].cuda()
        
                    optimizer.zero_grad()
                    map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs)
                    # map_x shape: batch,N,N
                    #pre_label = 0
                    
                    for ba in range(test_ba):
                        pre_label = 0
                        map_score = torch.sum(map_x[ba])/(32*32)
                        if map_score >= threshold:
                            pre_label = 1
                        if pre_label == spoof_label:
                            num += 1
                        if spoof_label!=1:
                            meanAcF.append(1-map_score.item())
                        else:
                            meanAcT.append(map_score.item())
                        # print(spoof_label,map_score)
                        
                    torch.cuda.empty_cache()


            with torch.no_grad():


        #rootDir = "D:/dataset/oulu/oulu/trainset/"
                val_data = Spoofing_test("/home/chang/dataset/B_face1/", transform=transforms.Compose([ToTensor(), Normaliztion()]))
                #val_data = Spoofing_train("../../../../devset", transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
                # val_data = Spoofing_valtest(image_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                test_ba = 1
                dataloader_val1 = DataLoader(val_data, batch_size=test_ba, shuffle=False, num_workers=4)
                
                # map_score_list = []

                num1 = 0
                for i, sample_batched in enumerate(dataloader_val1):
        #             # get the inputs
                    inputs, binary_mask, spoof_label = sample_batched['image_x'].cuda(), sample_batched['binary_mask'].cuda(), sample_batched['spoofing_label'].cuda() 
                    # inputs = sample_batched['image_x'].cuda()
                    # binary_mask = sample_batched['binary_mask'].cuda()
        
                    optimizer.zero_grad()
                    map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs)
                    # map_x shape: batch,N,N
                    #pre_label = 0
                    
                    for ba in range(test_ba):
                        pre_label = 0
                        map_score = torch.sum(map_x[ba])/(32*32)
                        if map_score >= threshold:
                            pre_label = 1
                        if pre_label == spoof_label:
                            num1 += 1
                        
                        # print(spoof_label,map_score)
                        
                    torch.cuda.empty_cache()
                

            
        #     # save the model until the next improvement  
            print("TP",np.mean(meanAcT))   
            print("TN",np.mean(meanAcF))  
            meanAcT = np.array(meanAcT)
            meanAcF = np.array(meanAcF)
            TP = len(meanAcT[meanAcT>threshold])
            TN = len(meanAcF[meanAcF>1 - threshold])
            acc = (TP+TN)/(len(meanAcF)+len(meanAcT))
            accuracy = num/len(dataloader_val)
            print("ACC",acc,":T ",TP,":",len(meanAcT)," F ",TN,":",len(meanAcF)) 
            print("ACCURACY:",accuracy)
            accuracy_list.append(accuracy)

            accuracy1 = num1/len(dataloader_val1)
            accuracy1_list.append(accuracy1)
            print("ACCURACY1:",accuracy1)

            log_file.write('val: TP= %.4f,:%.4f TN= %.4f,:%.4f ACC=%.4f \n' % (TP,len(meanAcT),TN,len(meanAcF),acc))
            log_file.write('ACCURACY = %.4f \n' % accuracy)
            log_file.write('ACCURACY1 = %.4f \n' % accuracy1)
            
            #print(args.log+'/'+args.log)
            torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pkl' % (epoch + 1))
            # break


    print('Finished Training')
    log_file.close()

    plt.plot(MSELoss_list, label = "MSELoss")
    plt.plot(Contrast_depth_loss_list, label = "depth_loss")
    plt.plot(total_loss_list, label = "total_loss")
    plt.title("loss")
    plt.legend()
    plt.show()

    plt.plot(accuracy_list, label = "accuracy")
    plt.plot(accuracy1_list, label = "accuracy1")
    #plt.plot(total_loss_list, label = "total_loss")
    plt.title("accuracy")
    plt.legend()
    plt.show()

  
 

if __name__ == "__main__":
    #mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  #default=0.0001
    parser.add_argument('--batchsize', type=int, default=5, help='initial batchsize')  #default=7
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 500  | DPC = 400
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--log', type=str, default="log121002", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')

    args = parser.parse_args()
    train_test()
