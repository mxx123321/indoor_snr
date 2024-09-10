from torch.utils.data import DataLoader
from dataloader import Dataset_SNR50,Dataset_SNR10,Dataset_SNR20
from utils.masking import Masking
from models.TestPos import TestPos
from utils.teacher import EMATeacher
from utils.sam import SAM
import torch
#alpha', default=0.999
#pseudo_label_weight = None
from utils.domain_discriminator import DomainDiscriminator
import torch.nn as nn
from utils.cdan import ConditionalDomainAdversarialLoss
from utils.metric import cal_average
from models.convnext import convnext_tiny
from torchvision.models import resnet
device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

Source_Dataloader = DataLoader(Dataset_SNR10(),32)
Target_Dataloader = DataLoader(Dataset_SNR20(),32)

Masking = Masking(1,0.001)
model = convnext_tiny().to(device)
teacher = EMATeacher(model,alpha=0.9,pseudo_label_weight=None)

base_optimizer = torch.optim.SGD
#base_optimizer = base_optimizer#.to(device)
# NEED CHANGE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
domain_discri = DomainDiscriminator(2304, hidden_size=256).to(device)

# Parameters NEED CHANGE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ad_optimizer = base_optimizer(domain_discri.get_parameters(), 0.001, momentum=0.9, weight_decay=1e-3, nesterov=True)
# optimizer = SAM(model.get_parameters(), base_optimizer, rho=0.02, adaptive=False,
#                     lr=0.001, momentum=0.9, weight_decay=1e-3, nesterov=True)

optimizer = SAM(model.parameters(), base_optimizer, rho=0.02, adaptive=False,
                    lr=0.001, momentum=0.9, weight_decay=1e-3, nesterov=True)
MSE_loss = nn.MSELoss().to(device)
domain_adv = ConditionalDomainAdversarialLoss(domain_discri,\
        entropy_conditioning=False,\
        num_classes=2, features_dim=5, \
            randomized=False,
        randomized_dim=5).to(device)
for epoch_this in range(1000):
    list_source = []
    for i, ((source, source_label), (target,_)) in enumerate(zip(Source_Dataloader, Target_Dataloader)):
        
        # (1) 获得source数据，source label，以及 target数据；三种数据

        source, source_label = source.float().to(device), source_label.float().to(device)
        target = target.float().to(device)
        i = i
        # (2) 获得masking后的target数据，以及，优化器的zero_grad()
        target_mask = Masking(target)
        optimizer.zero_grad()
        ad_optimizer.zero_grad()



        # (3) using teacher model to generate pseudo-label
        # NEED CHANGE parameters
        teacher.update_weights(model, 300*100 + i )
        pseudo_label_t, pseudo_prob_t = teacher(target)
        #print("111111")

        # (4) using source and target data to generate 真实的 和 虚伪的 预测
        x = torch.cat((source, target), dim=0)
        y, f = model(x) # predictions = self.head(f); predictions == y  f == f; final feature && output feature 

        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)
        #print("y_s, source_label",y_s, source_label)
        mse_loss = MSE_loss(y_s, source_label)

        # (5) 使用mask掉的输入，来预测mask的输出，不需要hidden feature
        y_target_masked, _ = model(target_mask)
        #print(y_target_masked.shape,pseudo_label_t.shape)
        masking_loss_value = MSE_loss(y_target_masked, pseudo_label_t)

        loss = 0.0001 * masking_loss_value + mse_loss
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # 第二个大步骤
        #========================================
        #========================================
        #========================================
        #========================================
        #========================================
        #========================================
        # 第二个大步骤
        # Calculate task loss and domain loss
        y, f = model(x) 
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        mse_loss = MSE_loss(y_s, source_label)
        y_t_masked, _ = model(target_mask)
        #domain_adv 是 一个局外的损失函数，直接用的别人成熟的接口。
        #y_s：source数据的预测结果， f_s是预测头之前的hidden feature
        #y_t：traget数据的预测结果， f_t是预测头之前的hidden feature

        #也就是说， 这个loss的组成，是 1：domain loss；  2：交叉熵loss，算的是mask的预测和虚伪的标签；
        # 3：下面，又算了分类的loss，也是和上面的一样的1：交叉熵loss，算的是source的预测和真实的标签；
        transfer_loss = domain_adv(y_s, f_s, y_t, f_t)  +   MSE_loss(y_t_masked, pseudo_label_t)
        
        
        # 这个是区分二元分类的准确度，也就是分： 1.source 2.target，做展示的，并不是真正的loss
        domain_acc = domain_adv.domain_discriminator_accuracy # 这个是区分二元分类的准确度，也就是分： 1.source 2.target，做展示的，并不是真正的loss
        loss = 0.0001 * transfer_loss + mse_loss

        loss.backward()
        ad_optimizer.step()
        # Update parameters (Sharpness-Aware update)
        optimizer.second_step(zero_grad=True)
        #print("分类准确度",domain_acc)
        this_aver_mae = cal_average(y_s,source_label)
        list_source.append(this_aver_mae.item())
    print("mean error",sum(list_source)/len(list_source))