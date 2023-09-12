
import torch.nn as nn
from Networks import InceptionV3_Network, Attention, Linear,residual_block
from torch import optim
import numpy as np
import torch
import time
import torch.nn.functional as F
import math
class MGRL_Model(nn.Module):
    def __init__(self, hp):
        super(MGRL_Model, self).__init__()

        self.backbone_network = InceptionV3_Network()
        self.backbone_train_params = self.backbone_network.parameters()

        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.attn_network = Attention()
        self.attn_network.apply(init_weights)
        self.attn_train_params = self.attn_network.parameters()

        self.linear_network = Linear(hp.feature_num)
        self.linear_network.apply(init_weights)
        self.linear_train_params = self.linear_network.parameters()

        self.block=residual_block()
        self.block.apply(init_weights)
        self.block_train_params=self.block.parameters()

        self.optimizer = optim.Adam([
            {'params': filter(lambda param: param.requires_grad, self.backbone_train_params), 'lr': hp.backbone_lr},
            {'params': self.attn_train_params, 'lr': hp.lr},
            {'params': self.linear_train_params, 'lr': hp.lr},
            {'params': self.block_train_params, 'lr': hp.lr}])

        self.loss = nn.TripletMarginLoss(margin=0.3)
        self.hp = hp
    
    def train_model(self, batch):
        self.train()
        positive_feature_local_batch=[]
        negative_feature_local_batch=[]
        sample_feature_local_batch=[]
        
        positive_feature_complete = self.linear_network(self.attn_network(self.backbone_network(batch['positive_img'].to(self.hp.device),1)))
        negative_feature_complete = self.linear_network(self.attn_network(self.backbone_network(batch['negative_img'].to(self.hp.device),1)))
        sample_feature_complete = self.linear_network(self.attn_network(self.backbone_network(batch['sketch_img'].to(self.hp.device),1)))
        index_4=[(torch.argwhere(data==1)).view(-1) for data in batch['bool_mat_4']]
        index_9=[(torch.argwhere(data==1)).view(-1) for data in batch['bool_mat_9']]

        positive_part4_detain=[torch.index_select(sketch_part,dim=0,index=index_4[i]) for i,sketch_part in enumerate(batch['positive_part4'])]
        positive_part9_detain=[torch.index_select(sketch_part,dim=0,index=index_9[i]) for i,sketch_part in enumerate(batch['positive_part9'])]
        negative_part4_detain=[torch.index_select(sketch_part,dim=0,index=index_4[i]) for i,sketch_part in enumerate(batch['negative_part4'])]
        negative_part9_detain=[torch.index_select(sketch_part,dim=0,index=index_9[i]) for i,sketch_part in enumerate(batch['negative_part9'])]
        sample_part4_detain=[torch.index_select(sketch_part,dim=0,index=index_4[i]) for i,sketch_part in enumerate(batch['sketch_part4'])]
        sample_part9_detain=[torch.index_select(sketch_part,dim=0,index=index_9[i]) for i,sketch_part in enumerate(batch['sketch_part9'])]

        loss_local=0
        loss_complete = self.loss(sample_feature_complete, positive_feature_complete, negative_feature_complete)
       
        for i in range(len(positive_part4_detain)):
            positive_feature_local=self.linear_network(self.attn_network(self.block(self.backbone_network(positive_part4_detain[i].to(self.hp.device),0))))
            negative_feature_local=self.linear_network(self.attn_network(self.block(self.backbone_network(negative_part4_detain[i].to(self.hp.device),0))))
            sample_feature_local=self.linear_network(self.attn_network(self.block(self.backbone_network(sample_part4_detain[i].to(self.hp.device),0))))
            loss_part = self.loss(sample_feature_local,positive_feature_local,negative_feature_local)
            loss_local = loss_part+loss_local

        for i in range(len(positive_part9_detain)):
            positive_feature_local=self.linear_network(self.attn_network(self.block(self.backbone_network(positive_part9_detain[i].to(self.hp.device),0))))
            negative_feature_local=self.linear_network(self.attn_network(self.block(self.backbone_network(negative_part9_detain[i].to(self.hp.device),0))))
            sample_feature_local=self.linear_network(self.attn_network(self.block(self.backbone_network(sample_part9_detain[i].to(self.hp.device),0))))
            loss_part = self.loss(sample_feature_local,positive_feature_local,negative_feature_local)

        loss=loss_complete+loss_local

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        return loss.item()

    def evaluate_NN(self, dataloader,a,b):
        self.eval()

        self.Sketch_Array_Test = []
        self.Image_Array_Test = []       
        Sketch_Feature_ALL_local =[]
        Image_Feature_ALL_local = []
        Sketch_exist_local=[]

        for idx, batch in enumerate(dataloader):
            if self.hp.condition:
                sketch_feature = self.cat_feature(self.attn_network(
                    self.backbone_network(batch['sketch_img'].to(self.hp.device))), batch['condition'].to(self.hp.device))
                positive_feature = self.linear_network(self.cat_feature(self.attn_network(
                    self.backbone_network(batch['positive_img'].to(self.hp.device))), batch['condition'].to(self.hp.device)))
                
            else:
                sample_feature_complete = self.attn_network(self.backbone_network(batch['sketch_img'].to(self.hp.device),1))
                positive_feature_complete = self.linear_network(self.attn_network(self.backbone_network(batch['positive_img'].to(self.hp.device),1)))[0]

                positive_feature_local_4 = [self.linear_network(self.attn_network(self.block(self.backbone_network((batch[0].view(1,batch.shape[1],batch.shape[2],batch.shape[3]).to(self.hp.device)),0)))) for batch in batch['positive_part4']] 
                sample_feature_local = [self.attn_network(self.block(self.backbone_network((batch.to(self.hp.device)),0))) for batch in batch['sketch_part']]
                positive_feature_local_9 = [self.linear_network(self.attn_network(self.block(self.backbone_network((batch[0].view(1,batch.shape[1],batch.shape[2],batch.shape[3]).to(self.hp.device)),0)))) for batch in batch['positive_part9']]                 

            self.Sketch_Array_Test.append(sample_feature_complete)
            self.Image_Array_Test.append(positive_feature_complete)

            sketch_local_pool=[]
            sketch_local_pool.extend(sample_feature_local)
            sketch_local_pool=torch.stack(sketch_local_pool)
            Sketch_Feature_ALL_local.append(sketch_local_pool)

            positive_local_pool=[]
            positive_local_pool.extend(positive_feature_local_4)
            positive_local_pool.extend(positive_feature_local_9)
            positive_local_pool=torch.stack(positive_local_pool)

            Image_Feature_ALL_local.append(positive_local_pool)


            exist_pool=[]
            exist_pool.extend(batch['bool_mat'])

            exist_pool=torch.stack(exist_pool)
            Sketch_exist_local.append(exist_pool)


        self.Sketch_Array_Test = torch.stack(self.Sketch_Array_Test).to(self.hp.device)
        self.Image_Array_Test = torch.stack(self.Image_Array_Test).to(self.hp.device)
        Sketch_Feature_ALL_local =torch.stack(Sketch_Feature_ALL_local).to(self.hp.device)     
        Image_Feature_ALL_local =torch.stack(Image_Feature_ALL_local).to(self.hp.device)
        Sketch_exist_local=torch.stack(Sketch_exist_local).to(self.hp.device)

        num_of_Sketch_Step = len(self.Sketch_Array_Test[0])
        avererage_area = []
        avererage_area_percentile = []
        avererage_ourB = []
        avererage_ourA = []

        exps = np.linspace(1,num_of_Sketch_Step, num_of_Sketch_Step) / num_of_Sketch_Step
        factor = np.exp(1 - exps) / np.e
        rank_all = torch.zeros(len(self.Sketch_Array_Test), num_of_Sketch_Step)
        rank_all_percentile = torch.zeros(len(self.Sketch_Array_Test), num_of_Sketch_Step)


        num = list(range(70))
        Xmin = np.min(num)
        Xmax = np.max(num)
        a = 0
        b = 1
        Atten_num = a + (b-a)/(Xmax-Xmin)*(num-Xmin)

        for i_batch, sanpled_batch in enumerate(self.Sketch_Array_Test):
            mean_rank = []
            mean_rank_percentile = []
            mean_rank_ourB = []
            mean_rank_ourA = []

            for i_sketch in range(sanpled_batch.shape[0]):

                sketch_feature_complete = self.linear_network(sanpled_batch[i_sketch].unsqueeze(0).to(self.hp.device))
                target_distance_complete = F.pairwise_distance(sketch_feature_complete.to(self.hp.device), self.Image_Array_Test[i_batch].unsqueeze(0).to(self.hp.device))

                distance_complete = F.pairwise_distance(sketch_feature_complete.to(self.hp.device), self.Image_Array_Test.to(self.hp.device))

                part_exist=Sketch_exist_local[i_batch,0:13,i_sketch]
                part_exist_4=Sketch_exist_local[i_batch,:4,i_sketch]
                part_exist_9=Sketch_exist_local[i_batch,4:13,i_sketch]
                num_4=np.array(part_exist_4.cpu()).sum()
                num_9=np.array(part_exist_9.cpu()).sum()
                part_index=torch.argwhere(part_exist==1).view(-1).to(self.hp.device)

                sketch_part_feature_detain=torch.index_select(Sketch_Feature_ALL_local[i_batch,0:13,i_sketch,:],dim=0,index=part_index).to(self.hp.device)

                sketch_part_feature_detain=F.normalize(self.linear_network(sketch_part_feature_detain.unsqueeze(0).to(self.hp.device)).squeeze(0))
                positive_part_feature_detain_target=torch.index_select(Image_Feature_ALL_local[i_batch,0:13,:],dim=0,index=part_index).squeeze(1).to(self.hp.device)
                target_distance_part = F.pairwise_distance(sketch_part_feature_detain, positive_part_feature_detain_target)
                target_distance_4_mean=torch.sum(target_distance_part[:num_4])/num_4

                target_distance_9_mean=torch.sum(target_distance_part[num_4:num_9+num_4])/num_9

                positive_part_feature_detain=torch.index_select(Image_Feature_ALL_local[:,:13,:,:],dim=1,index=part_index).to(self.hp.device).squeeze(2)
                distance_part=F.pairwise_distance(sketch_part_feature_detain.to(self.hp.device),positive_part_feature_detain.to(self.hp.device))
                distance_4_mean=torch.sum(distance_part[:,0:num_4],dim=1)/num_4
                distance_9_mean=torch.sum(distance_part[:,num_4:num_9+num_4],dim=1)/num_9

                Attention_num = round(Atten_num[i_sketch], 2)

                if self.hp.distance_select =='com+part4_decned+part9_decend':
                    target_distance = target_distance_complete+round(math.exp(-(0.1*Attention_num)), 2)*target_distance_4_mean+round(math.exp(-(0.1*Attention_num)), 2)*target_distance_9_mean
                    distance = distance_complete+round(math.exp(-(0.1*Attention_num)), 2)*distance_4_mean+round(math.exp(-(0.1*Attention_num)), 2)*distance_9_mean
                elif self.hp.distance_select =='com+part4_decned':
                    target_distance = target_distance_complete+round(math.exp(-(0.1*Attention_num)), 2)*target_distance_4_mean
                    distance = distance_complete+round(math.exp(-(0.1*Attention_num)), 2)*distance_4_mean
                elif self.hp.distance_select =='com':
                    target_distance = target_distance_complete
                    distance = distance_complete
                elif self.hp.distance_select =='com_1+part4_1+part9_1':
                    target_distance = target_distance_complete+target_distance_4_mean+target_distance_9_mean
                    distance = distance_complete+distance_4_mean+distance_9_mean
                elif self.hp.distance_select =='com_1+part4_+part9_':
                    target_distance = target_distance_complete+a*target_distance_4_mean+b*target_distance_9_mean
                    distance = distance_complete+a*distance_4_mean+b*distance_9_mean
                elif self.hp.distance_select =='com_1+part4_1+part9_decend':
                    target_distance = target_distance_complete+target_distance_4_mean+round(math.exp(-(0.1*Attention_num)), 2)*target_distance_9_mean
                    distance = distance_complete+distance_4_mean+round(math.exp(-(0.1*Attention_num)), 2)*distance_9_mean
                elif self.hp.distance_select =='com_1+part4_decend+part9_1':
                    target_distance = target_distance_complete+round(math.exp(-(0.1*Attention_num)), 2)*target_distance_4_mean+target_distance_9_mean
                    distance = distance_complete+round(math.exp(-(0.1*Attention_num)), 2)*distance_4_mean+distance_9_mean

                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()

                rank_all_percentile[i_batch, i_sketch] = (len(distance) - rank_all[i_batch, i_sketch]) / (len(distance) - 1)
                if rank_all[i_batch, i_sketch].item() == 0:

                    mean_rank.append(1.)
                else:
                    mean_rank.append(1/rank_all[i_batch, i_sketch].item())
                    mean_rank_percentile.append(rank_all_percentile[i_batch, i_sketch].item())
                    mean_rank_ourB.append(1/rank_all[i_batch, i_sketch].item() * factor[i_sketch])
                    mean_rank_ourA.append(rank_all_percentile[i_batch, i_sketch].item()*factor[i_sketch])

            avererage_area.append(np.sum(mean_rank)/len(mean_rank))
            avererage_area_percentile.append(np.sum(mean_rank_percentile)/len(mean_rank_percentile))
            avererage_ourB.append(np.sum(mean_rank_ourB)/len(mean_rank_ourB))
            avererage_ourA.append(np.sum(mean_rank_ourA)/len(mean_rank_ourA))

        print(rank_all)
        print('MB', list(np.sum(np.array(1 / rank_all), axis=0) / len(rank_all)))
        print('MA', list(np.sum(np.array(rank_all_percentile), axis=0)/ len(rank_all)))
        print('wMB', list(np.sum(np.array(1 / rank_all), axis=0) / len(rank_all)*factor))
        print('wMA', list(np.sum(np.array(rank_all_percentile), axis=0)/ len(rank_all)*factor))
        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]
        #A@1 A@5 A%10
        meanIOU = np.mean(avererage_area)
        meanMA = np.mean(avererage_area_percentile)
        meanOurB = np.mean(avererage_ourB)
        meanOurA = np.mean(avererage_ourA)

        return top1_accuracy, top5_accuracy, top10_accuracy, meanIOU, meanMA, meanOurB, meanOurA

    def SortNameByData(self, dataList, nameList):
        convertDic = {}
        sortedDic = {}
        sortedNameList = []
        for index in range(len(dataList)):
            convertDic[index] = dataList[index]
        sortedDic = sorted(convertDic.items(), key=lambda item: item[1], reverse=False)
        for key, _ in sortedDic:
            sortedNameList.append(nameList[key])
        return sortedNameList