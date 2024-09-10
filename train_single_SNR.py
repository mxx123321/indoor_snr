# train on a single SNR and save the pth

from torch.utils.data import DataLoader
from dataloader import Dataset_SNR50, Dataset_SNR10, Dataset_SNR20
from utils.masking import Masking
from models.TestPos import TestPos
from utils.teacher import EMATeacher
from utils.sam import SAM
import torch
import os
import datetime
import argparse
from models.resnet import resnet18_pos
# alpha', default=0.999
from models.xjbLink import MultiChannelModel
# pseudo_label_weight = None
from utils.domain_discriminator import DomainDiscriminator
import torch.nn as nn
from utils.cdan import ConditionalDomainAdversarialLoss
from utils.metric import cal_average, cal_averageRMSE,best_position
from models.convnext import convnext_tiny
from models.convmixer import ConvMixer
import torch.optim as optim
from models.vision_transformer import vit_b_16_pos
#
from utils.set_seed import set_seed
set_seed(42)
#
from models.xjbLink import MultiChannelModel

#target_aug
parser = argparse.ArgumentParser(description='Train different models.')
parser.add_argument('--model', type=str, default='vit', help='Choose the model to train: v32, r18, r34, r50, r101, r152')
parser.add_argument('--device', type=str, default='cuda:0', help='Choose the model to train: v32, r18, r34, r50, r101, r152')
parser.add_argument('--batchsize', type=int, default=64, help='Choose the model to train: v32, r18, r34, r50, r101, r152')

parser.add_argument('--target_aug', type=str, default='noise', help='masking,noise')
parser.add_argument('--noise_weight', type=float, default=0.0001, help='noise')

parser.add_argument('--source_aug', type=str, default='None', help='masking,noise')
parser.add_argument('--source_noise_weight', type=float, default=0.0001, help='noise')

parser.add_argument('--loss_1_weight', type=float, default=0.1, help='masking,noise')
parser.add_argument('--loss_2_weight', type=float, default=0.1, help='masking,noise')

parser.add_argument('--mask_patch', type=int, default=1, help='masking,noise')
parser.add_argument('--mask_ratio', type=float, default=0.1, help='masking,noise')
#args.using_diffusion
parser.add_argument('--using_diffusion', type=str, default='True', help='masking,noise')
parser.add_argument('--train_setting', type=str, default='10_50', help='masking,noise')


args = parser.parse_args()

# 设备初始化
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
Dataset_source =  eval('Dataset_SNR' + args.train_setting[:2] + '()')
Dataset_target =  eval('Dataset_SNR' + args.train_setting[-2:] + '()')
# 初始化源数据集和目标数据集的数据加载器
source_dataloader = DataLoader(Dataset_source, batch_size=args.batchsize,shuffle=True)
target_dataloader = DataLoader(Dataset_target, batch_size=args.batchsize,shuffle=True)

# 初始化遮蔽、模型和教师模型
if args.target_aug == "masking":
    masking = Masking(args.mask_patch, args.mask_ratio)

if args.model == 'vit':
    model = vit_b_16_pos(num_classes=3).to(device)
elif args.model == 'resnet':
    from models.resnet import resnet18_pos
    model = resnet18_pos().to(device)
#model = MultiChannelModel()
elif args.model == 'MultiChannelModel':
    #from models.resnet import resnet18_pos
    model = MultiChannelModel().to(device)
#FullModel
elif args.model == 'MPRI':
    from models.Junk import FullModel
    model = FullModel().to(device) 

#model = ConvMixer(5,128,4).to(device)
teacher = EMATeacher(model, alpha=0.1, pseudo_label_weight=None)

# 定义基础优化器和域鉴别器
# base_optimizer = torch.optim.SGD
base_optimizer = torch.optim.Adam
# base_optimizer = optim.Adam(model.parameters(),lr=0.001)

# 定义模型和域鉴别器的优化器
# optimizer = SAM(model.parameters(), base_optimizer, rho=0.02, adaptive=False, lr=0.002, momentum=0.9, weight_decay=1e-3, nesterov=True)
# optimizer = SAM(model.parameters(), base_optimizer,  lr=0.001)
# optimizer = SAM(model.parameters(), base_optimizer,  lr=0.001)
optimizer = base_optimizer(model.parameters(), lr=0.001)
# 定义损失函数
mse_loss = nn.MSELoss().to(device)

# 训练循环
for epoch in range(200):
    print("第几个epoch",epoch)
    start_time = datetime.datetime.now()
    model.train()  # 设置模型为训练模式
  # 设置模型为训练模式
    list_source_errors = []
    
    list_domain_acc = []
    for i, (source, source_label) in enumerate(source_dataloader):
        source, source_label = source.float().to(device), source_label.float().to(device)
        
        
        
        # 梯度清零
        optimizer.zero_grad()
   
        
        # 更新教师模型并生成伪标签
        predictions,_ = model(source)#64,3
       
        # pred_source = model(source)
        # print(pred_source.shape)
        # print(source_label.shape)
        # 计算源数据的MSE损失
        loss_mse_source = mse_loss(predictions, source_label)
        
        
        # loss = loss_mse_source
        # print(loss)
        # base_optimizer.zero_grad()
        loss_mse_source.backward()
        optimizer.step()
        
        
        # 计算源数据预测的平均绝对误差
        mean_absolute_error = cal_average(predictions, source_label)
        #source_label
        #pred_source
        list_source_errors.append(mean_absolute_error.item())
        
    end_time = datetime.datetime.now()
    # 计算程序运行时间
    duration = end_time - start_time
    #print("程序运行时间:", duration)
    #print("0-0-0-0-0-0-0-")
    with open("/root/mxx_code/MIC_transfer_pos_code/cloud_mask_transfer_MIC_code/single_SNR_output_file/output_{}.txt".format(args.model), "a") as file:
        file.write("Epoch is : " + str(epoch) + "\n")
        file.write("Time is : " + str(duration) + "\n")
        file.write("Source Mean error: " + str(sum(list_source_errors) / len(list_source_errors)) + "\n")
list_target_errors = []
list_target_errors_RMSE = []
list_target = []
list_single_error = []
#mean_absolute_error_RMSE list_target_erroes
model.eval()
for i, (source, source_label) in enumerate(target_dataloader):
    source, source_label = source.float().to(device), source_label.float().to(device)
    
    
    
    # 梯度清零
 
   
    
    # 更新教师模型并生成伪标签
    predictions,_ = model(source)#64,3
    
    # pred_source = model(source)
    # print(pred_source.shape)
    # print(source_label.shape)
    # 计算源数据的MSE损失
    loss_mse_source = mse_loss(predictions, source_label)
    
    
    # loss = loss_mse_source
    # print(loss)
    # base_optimizer.zero_grad()
    
    
    
    
    # 计算源数据预测的平均绝对误差
    mean_absolute_error = cal_average(predictions, source_label)
    #cal_averageRMSE
    mean_absolute_error_RMSE = cal_averageRMSE(predictions, source_label)
    
    #source_label
    #pred_source
    list_target_errors.append(mean_absolute_error.item())
    print("len list_target_errors",len(list_target_errors))
    list_target_errors_RMSE.append(mean_absolute_error_RMSE.item())
    #
    # 
    #print(best_position(predictions, source_label))
    list_single_error.append(best_position(predictions, source_label).tolist())
    
    list_target.append(source_label.tolist())
print(list_single_error)    
with open("/root/mxx_code/MIC_transfer_pos_code/cloud_mask_transfer_MIC_code/single_SNR_output_file/output_{}.txt".format(args.model), "a") as file:
        file.write("Testing is : " + str(epoch) + "\n")
        file.write("Testing Mean error: " + str(sum(list_target_errors) / len(list_target_errors)) + "\n")
        file.write("RMSE Mean error: " + str(sum(list_target_errors_RMSE) / len(list_target_errors_RMSE)) + "\n")


with open("/root/mxx_code/MIC_transfer_pos_code/cloud_mask_transfer_MIC_code/single_SNR_output_file/output_all_{}.txt".format(args.model), "a") as file:
        file.write("Setting is : " + str(args) + "\n")
        file.write("Target Mean error: " + str(sum(list_target_errors) / len(list_target_errors)) + "\n")
        file.write("RMSE Mean error: " + str(sum(list_target_errors_RMSE) / len(list_target_errors_RMSE)) + "\n")
        
        
#----------------
#----------------
#----------------
folder_path = "/root/mxx_code/MIC_transfer_pos_code/cloud_mask_transfer_MIC_code/single_SNR_output_file/"
os.makedirs(folder_path, exist_ok=True)

file_path = folder_path + "single_{}.text".format(args.model)

with open(file_path, "a+") as fp:
    print(args.train_setting, file=fp)
    print("平均值为：%f" % (sum([sum(row) for row in list_single_error]) / sum(len(row) for row in list_single_error) ), file=fp)
    print("单个精度", list_single_error, file=fp)
    print("对应位置", list_target, file=fp)
    
#sum([sum(row) for row in data]) / sum(len(row) for row in data)
    #print("平均值为：%f" % (sum([sum(row) for row in list_single_error]) / sum(len(row) for row in list_single_error) ), file=fp)

    #print("rmse为：%f" % torch.mean(torch.stack(list_single_error[:-1])), file=fp)

    #print("方差为：%f" % torch.var(torch.stack(list_single_error)), file=fp)
    #print("标准差为:%f" % torch.std(torch.stack(list_single_error)), file=fp)
    #single_train = []
    # single_val = []
    # single_test = []
fp.close()