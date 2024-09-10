from torch.utils.data import DataLoader
from dataloader import Dataset_SNR50, Dataset_SNR10, Dataset_SNR20

from models.TestPos import TestPos

import torch
import datetime
import argparse
from models.resnet import resnet18_pos
# alpha', default=0.999
# pseudo_label_weight = None

from utils.cdan import ConditionalDomainAdversarialLoss
from utils.metric import cal_average, cal_averageRMSE, best_position
from utils.set_seed import set_seed
from utils.domain_discriminator import DomainDiscriminator
from utils.teacher import EMATeacher
from utils.sam import SAM
from utils.masking import Masking
from utils.masking import MultiMasking

from models.convnext import convnext_tiny
from models.convmixer import ConvMixer
from models.xjbLink import MultiChannelModel

import torch.optim as optim
from models.vision_transformer import vit_b_16_pos#(num_classes=3)
# 设定随机种子
import torch.nn as nn
set_seed(42)
#

#
#target_aug noise_weight source_aug source_noise_weight
parser = argparse.ArgumentParser(description='Train different models.')
parser.add_argument('--model', type=str, default='vit', help='Choose the model to train: v32, r18, r34, r50, r101, r152')
parser.add_argument('--device', type=str, default='cuda:0', help='Choose the model to train: v32, r18, r34, r50, r101, r152')
parser.add_argument('--batchsize', type=int, default=64, help='Choose the model to train: v32, r18, r34, r50, r101, r152')

parser.add_argument('--target_aug', type=str, default='masking', help='masking,noise')
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
    masking = MultiMasking(args.mask_patch, args.mask_ratio)
masking = MultiMasking(args.mask_patch, args.mask_ratio)
#
if args.model == 'vit':
    model = vit_b_16_pos(num_classes=3).to(device)
    print(model)
elif args.model == 'resnet':
    #from models.resnet import resnet18_pos
    model = resnet18_pos().to(device)
#model = MultiChannelModel()
elif args.model == 'MultiChannelModel':
    #from models.resnet import resnet18_pos
    model = MultiChannelModel().to(device)
elif args.model == 'MPRI':
    from models.Junk import FullModel
    model = FullModel().to(device) 

#model = ConvMixer(5,128,4).to(device)
teacher = EMATeacher(model, alpha=0.1, pseudo_label_weight=None)

# 定义基础优化器和域鉴别器
# base_optimizer = torch.optim.SGD
base_optimizer = torch.optim.Adam
# base_optimizer = optim.Adam(model.parameters(),lr=0.001)
if args.model == 'resnet':
    domain_discriminator = DomainDiscriminator(1536, hidden_size=256).to(device) # #128*3
elif args.model == 'MultiChannelModel':
    domain_discriminator = DomainDiscriminator(1536, hidden_size=256).to(device) # #128*3
elif args.model == 'MPRI':
    domain_discriminator = DomainDiscriminator(768, hidden_size=256).to(device) # #128*3
else:
    domain_discriminator = DomainDiscriminator(768, hidden_size=256).to(device) # #128*3
# 定义模型和域鉴别器的优化器
ad_optimizer = base_optimizer(domain_discriminator.parameters(), lr=0.001)
# optimizer = SAM(model.parameters(), base_optimizer, rho=0.02, adaptive=False, lr=0.002, momentum=0.9, weight_decay=1e-3, nesterov=True)
#optimizer = SAM(model.parameters(), base_optimizer,  lr=0.001)

# optimizer = base_optimizer(model.parameters(),  lr=0.001)
optimizer = SAM(model.parameters(), base_optimizer,  lr=0.001)


# 定义损失函数
mse_loss = nn.MSELoss().to(device)
domain_adv_loss = ConditionalDomainAdversarialLoss(domain_discriminator, entropy_conditioning=False, num_classes=2, features_dim=5, randomized=False, randomized_dim=5).to(device)

# 训练循环
for epoch in range(500):
    print("第几个epoch",epoch)
    start_time = datetime.datetime.now()
    model.train()  # 设置模型为训练模式
    domain_discriminator.train()  # 设置模型为训练模式
    list_source_errors = []
    list_target_errors = []
    list_source_errors_RMSE = []
    #
    list_target = []
    list_single_error = []
    
    
    list_domain_acc = []
    for i, ((source, source_label), (target, target_label)) in enumerate(zip(source_dataloader, target_dataloader)):
        source, source_label = source.float().to(device), source_label.float().to(device)
        target, target_label = target.float().to(device), target_label.float().to(device)
        
        #source 用不用增强
        if args.source_aug == 'noise':
            if args.using_diffusion == 'True':
                futher_weight = epoch
            elif args.using_diffusion == 'False':
                futher_weight = 1.0
            source = args.source_noise_weight * futher_weight * torch.randn_like(source) + source
        else:
            source = source

        # 对目标数据应用遮蔽
        if args.target_aug == "noise":
            if args.using_diffusion == 'True':
                futher_weight = epoch
            elif args.using_diffusion == 'False':
                futher_weight = 1.0
            target_masked = args.noise_weight * futher_weight * torch.randn_like(target) + target
            
            target_masked = masking(target)
            
        elif args.target_aug == "masking":
            target_masked = masking(target)
        else:
            target_masked = target
        
        # 梯度清零
        optimizer.zero_grad()
        ad_optimizer.zero_grad()
        
        # 更新教师模型并生成伪标签
        if epoch == 0:
            teacher.update_weights(model, 0)
        teacher.update_weights(model, epoch * len(source_dataloader) + i)
        pseudo_label_target, pseudo_prob_target = teacher(target)
        
        # 对源数据和目标数据进行前向传播
        combined_data = torch.cat((source, target), dim=0) #64,5,16,193
        predictions, features = model(combined_data)#64,3
        pred_source, pred_target = predictions.chunk(2, dim=0)  #分割成两个块
        feat_source, feat_target = features.chunk(2, dim=0)
        
        # pred_source = model(source)
        # print(pred_source.shape)
        # print(source_label.shape)
        # 计算源数据的MSE损失
        loss_mse_source = mse_loss(pred_source, source_label)
        
        # 对遮蔽的目标数据进行前向传播
        pred_target_masked, _ = model(target_masked)
        loss_masking = mse_loss(pred_target_masked, pseudo_label_target)
        
        # loss_masking 是 伪label的预测 和 mask后的 预测的label
        loss = args.loss_1_weight * loss_masking + loss_mse_source
        # loss = loss_mse_source
        # print(loss)
        # base_optimizer.zero_grad()
        loss.backward()
        # optimizer.step()
        optimizer.first_step(zero_grad=True)
        
        # 优化器的第二步（Sharpness-Aware更新）
        predictions, features = model(combined_data)
        #print(features.shape,"features")
        pred_source, pred_target = predictions.chunk(2, dim=0)
        feat_source, feat_target = features.chunk(2, dim=0)
        
        loss_mse_source = mse_loss(pred_source, source_label)
        # `pred_target_masked` is the prediction made by the model on the target data after applying a
        # specific augmentation technique, such as noise addition or masking. This prediction is used
        # to calculate the loss between the predicted labels and the pseudo labels generated by the
        # teacher model for the target data. The model is trained to minimize this loss, which helps
        # in improving the model's performance on the target domain by leveraging the pseudo labels
        # and the augmented target data.
        pred_target_masked, _ = model(target_masked)

        # print(feat_source.shape)

        transfer_loss = domain_adv_loss(pred_source, feat_source, pred_target, feat_target) + mse_loss(pred_target_masked, pseudo_label_target)
        #transfer_loss = 0
        domain_acc = domain_adv_loss.domain_discriminator_accuracy
    
        # 计算最终损失并更新域对抗优化器
        loss = args.loss_2_weight * transfer_loss + loss_mse_source
        # print(loss)
        loss.backward()
        ad_optimizer.step()
        # base_optimizer.step()
        # optimizer.step()
        optimizer.second_step(zero_grad=True)
        
        # 计算源数据预测的平均绝对误差
        mean_absolute_error = cal_average(pred_target, target_label)
        mean_absolute_error_source = cal_average(pred_source, source_label)
        #source_label
        #
        #
        
        mean_absolute_error_source_rmse = cal_averageRMSE(pred_target, target_label)
        #pred_source
        
        list_source_errors.append(mean_absolute_error_source.item())
        list_target_errors.append(mean_absolute_error.item())
        #
        list_source_errors_RMSE.append(mean_absolute_error_source_rmse.item())
        
        list_domain_acc.append(domain_acc.item())
        #
        #print(best_position(predictions, source_label))
        
        list_single_error.append(best_position(pred_target, target_label).tolist())
        
        list_target.append(source_label.tolist())
        
        
    end_time = datetime.datetime.now()
    # 计算程序运行时间
    duration = end_time - start_time
    #print("程序运行时间:", duration)
    
    with open("/root/mxx_code/MIC_transfer_pos_code/cloud_mask_transfer_MIC_code/output_file/output_{}.txt".format(args.model), "a") as file:
        file.write("Setting is : " + str(args) + "\n")
        file.write("Epoch is : " + str(epoch) + "\n")
        file.write("Time is : " + str(duration) + "\n")
        file.write("Source Mean error: " + str(sum(list_source_errors) / len(list_source_errors)) + "\n")
        file.write("Target Mean error: " + str(sum(list_target_errors) / len(list_target_errors)) + "\n")
        file.write("Domain Acc: " + str(sum(list_domain_acc) / len(list_domain_acc)) + "\n\n")
        # list_source_errors_RMSE
        file.write("Target RMSE Mean error: " + str(sum(list_source_errors_RMSE) / len(list_source_errors_RMSE)) + "\n")
    
    if epoch >= 490:
        with open("/root/mxx_code/MIC_transfer_pos_code/cloud_mask_transfer_MIC_code/output_file/output_all_{}.txt".format(args.model), "a") as file:
            file.write("Setting is : " + str(args) + "\n")
            file.write("Target Mean error: " + str(sum(list_target_errors) / len(list_target_errors)) + "\n")
            #
            file.write("Target RMSE Mean error: " + str(sum(list_source_errors_RMSE) / len(list_source_errors_RMSE)) + "\n")
            
            
#-0-====----------------------------

#----------------
#----------------
import os
#----------------
folder_path = "/root/mxx_code/MIC_transfer_pos_code/cloud_mask_transfer_MIC_code/output_file/"
os.makedirs(folder_path, exist_ok=True)

file_path = folder_path + "single_{}.text".format(args.model)

with open(file_path, "a+") as fp:
    print(args.train_setting, file=fp)
    print("平均值为：%f" % (sum([sum(row) for row in list_single_error]) / sum(len(row) for row in list_single_error) ), file=fp)
    print("单个精度", list_single_error, file=fp)
    print("对应位置", list_target, file=fp)
#sum([sum(row) for row in data]) / sum(len(row) for row in data)
    

    #print("rmse为：%f" % torch.mean(torch.stack(list_single_error[:-1])), file=fp)

    #print("方差为：%f" % torch.var(torch.stack(list_single_error)), file=fp)
    #print("标准差为:%f" % torch.std(torch.stack(list_single_error)), file=fp)
    #single_train = []
    # single_val = []
    # single_test = []
fp.close()