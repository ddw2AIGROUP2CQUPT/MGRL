import torch
import time
from model import MGRL_Model
from dataset_train import get_dataloader
import argparse

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MGRL Model')
    parser.add_argument('--dataset_name', type=str, default='Face-450', help='Face-1000 / Face-450')
    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--nThreads', type=int, default=4)
    parser.add_argument('--backbone_lr', type=float, default=0.00005)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--print_freq_iter', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--feature_num', type=int, default=8)
    parser.add_argument('--condition', type=int, default=0)
    parser.add_argument('--distance_select',type=str,default='com+part4+part9')
    hp = parser.parse_args()
    tb_logdir = r"./run/"

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
    dataloader_Train = get_dataloader(hp)
    print(hp)
    tb_writer = SummaryWriter(log_dir=tb_logdir)
    model = MGRL_Model(hp)
    model.to(hp.device)
    step_count, top1, top5, top10, top50, top100 = -1, 0, 0, 0, 0, 0
    mean_IOU_buffer = 0
    real_p = [0, 0, 0, 0, 0, 0]

    for i_epoch in range(hp.max_epoch):
        for batch_data in dataloader_Train:
            step_count = step_count + 1
            start = time.time()
            model.train()
            loss = model.train_model(batch=batch_data)
            tb_writer.add_scalars('loss',{'loss':loss}, step_count)
            if  step_count % hp.eval_freq_iter==0 and  int(step_count / hp.eval_freq_iter)>50:
                print('Epoch: {},Iteration: {},Loss:{:.8f}'.format(i_epoch,step_count,loss))
                torch.save(model.backbone_network.state_dict(),
                                    hp.dataset_name + '_f' + str(hp.feature_num) +'_' +str(int(step_count / hp.eval_freq_iter)) + '_backbone.pth')
                torch.save(model.attn_network.state_dict(),
                                    hp.dataset_name + '_f' + str(hp.feature_num) +'_' +str(int(step_count / hp.eval_freq_iter)) + '_attn.pth')
                torch.save(model.linear_network.state_dict(),
                                    hp.dataset_name + '_f' + str(hp.feature_num)  +'_'+str(int(step_count / hp.eval_freq_iter)) + '_linear.pth')
                torch.save(model.block.state_dict(),
                                    hp.dataset_name + '_f' + str(hp.feature_num) +'_' +str(int(step_count / hp.eval_freq_iter)) + '_block.pth')             
