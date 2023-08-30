from turtle import distance
import torch
import time
from model import SLFIR_Model
import os
from dataset_test import get_dataloader
import argparse
import torch.nn as nn
from Networks import InceptionV3_Network, Attention, Linear,residual_block
from torch import optim
import numpy as np
import torch
import time
import torch.nn.functional as F
import math
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":     
    parser = argparse.ArgumentParser(description='SLFIR Model')
    parser.add_argument('--dataset_name', type=str, default='Face-1000', help='Face-1000 / Face-450')
    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--nThreads', type=int, default=4)
    parser.add_argument('--backbone_lr', type=float, default=0.0005)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--print_freq_iter', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--feature_num', type=int, default=16)
    parser.add_argument('--condition', type=int, default=0)
    parser.add_argument('--distance_select',type=str,default='com_1+part4_1+part9_1')
    hp = parser.parse_args()
    if hp.dataset_name == 'Face-1000':
        hp.batchsize = 32
        hp.eval_freq_iter = 50
        hp.backbone_lr = 0.0005
        hp.lr = 0.005    
    elif hp.dataset_name == 'Face-450':
        hp.batchsize = 32
        hp.eval_freq_iter = 20
        hp.backbone_lr = 0.00005
        hp.lr = 0.0005

    if hp.condition:
        hp.condition_num = 10
    else:
        hp.condition_num = 0

    hp.device = torch.device("cuda:"+str(hp.gpu_id) if torch.cuda.is_available() else "cpu")
    dataloader_Test = get_dataloader(hp)
    print(hp)
    model = SLFIR_Model(hp)
    model.to(hp.device)
    mean_IOU_buffer = 0
    real_p = [0, 0, 0, 0, 0, 0]
    model_root_dir=os.path.join('/model/')
    model.backbone_network.load_state_dict(torch.load(model_root_dir+hp.dataset_name+'_f'+str(hp.feature_num)+'_best'+'_backbone.pth')) 
    model.attn_network.load_state_dict(torch.load(model_root_dir+hp.dataset_name+'_f'+str(hp.feature_num)+'_best'+'_attn.pth'))
    model.linear_network.load_state_dict(torch.load(model_root_dir+hp.dataset_name+'_f'+str(hp.feature_num)+'_best'+'_linear.pth'))
    model.block.load_state_dict(torch.load(model_root_dir+hp.dataset_name+'_f'+str(hp.feature_num)+'_best'+'_block.pth'))

    with torch.no_grad():
        start_time = time.time()
        top1, top5, top10, mean_IOU, mean_MA, mean_OurB, mean_OurA = model.evaluate_NN(dataloader_Test)
        print("TEST A@1: {}".format(top1))
        print("TEST A@5: {}".format(top5))
        print("TEST A@10: {}".format(top10))
        print("TEST M@B: {}".format(mean_IOU))
        print("TEST M@A: {}".format(mean_MA))
        print("TEST OurB: {}".format(mean_OurB))
        print("TEST OurA: {}".format(mean_OurA))
        print("TEST Time: {}".format(time.time()-start_time))
    if mean_IOU > mean_IOU_buffer:
        mean_IOU_buffer = mean_IOU
        real_p = [top1, top5, top10, mean_MA, mean_OurB, mean_OurA]
        print('Model Upgrate')
    print('REAL performance: Top1: {}, Top5: {}, Top10: {}, MB: {}, MA: {}, wMB: {}, wMA: {},'.format(real_p[0], real_p[1],
                                                                                                            real_p[2],
                                                                                                            mean_IOU_buffer,
                                                                                                            real_p[3],
                                                                                                            real_p[4],
                                                                                                            real_p[5]))