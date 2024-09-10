import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter

class ReshapeLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReshapeLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if out_channels < in_channels:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.reshape_operation = self.upsample
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.reshape_operation = self.pool

    def forward(self, x):
        x = self.conv(x)  # 调整通道数
        x = self.reshape_operation(x)  # 上采样或下采样
        return x

class MultiChannelModel(nn.Module):
    def __init__(self):
        super(MultiChannelModel, self).__init__()

        # 通用卷积-BN-ReLU组合
        def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
                # 通用卷积-BN-ReLU组合
        def bn_relu_conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
            return nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            )
        # 第一个层（Original Feature）
        self.original_feature = nn.Sequential(
            conv_bn_relu(5, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
        )
        self.ori_process_layer = nn.ModuleList([
            nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=1, padding=(0, 1))
            )for _ in range(3) 
        ])

        # 第二个层（Half Feature）
        self.half_feature = nn.Sequential(
            conv_bn_relu(5, 256, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(2),
        )
        self.half_process_layer = nn.ModuleList([
            nn.Sequential(
            bn_relu_conv(256, 256, kernel_size=(3,3), padding=1)
            )for _ in range(3) 
        ])
        # 第三个层（Quarter Feature）
        self.quarter_feature = nn.Sequential(
            conv_bn_relu(5, 512, kernel_size=3, stride=4, padding=1),
            nn.MaxPool2d(2),
        )
        self.quarter_process_layer =  nn.ModuleList([
            nn.Sequential(
            bn_relu_conv(512, 512, kernel_size=(3, 1), padding=(1, 0))
            )for _ in range(3) 
        ])
       #torch.Size([1, 512, 2, 24])
        self.final_layer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 添加全局平均池化层
            nn.Flatten(),  # 展平成一维
            nn.Linear(512, 3)  # 全连接层，从512维映射到3维
        )
        # self.up64_128 = nn.ModuleList([
        #        ReshapeLayer(64,128)
        #        for _ in range(3) 
        # ])
        self.up128_256 =  nn.ModuleList([
            ReshapeLayer(128,256)
              for _ in range(3) 
        ])
        self.up256_512 = nn.ModuleList([
            ReshapeLayer(256,512)
              for _ in range(4) 
        ])
        self.up128_512 = nn.Sequential(
            ReshapeLayer(128,256),
            ReshapeLayer(256,512)
        )
     

        self.down512_256 = nn.ModuleList([
            ReshapeLayer(512,256)
              for _ in range(3) 
        ])
        self.down256_128 = nn.ModuleList([
            ReshapeLayer(256,128)
            for _ in range(3) 
        ])
        # self.down128_64 = nn.ModuleList([
        #     ReshapeLayer(128,64)
        #     for _ in range(3) 
        # ])


    def forward(self, x):
        original_feature = self.original_feature(x)
        half_feature = self.half_feature(x)
        quarter_feature = self.quarter_feature(x)

        for i in range(3):  # 三次迭代
            original_process = self.ori_process_layer[i](original_feature)
            half_process = self.half_process_layer[i](half_feature)
            quarter_process = self.quarter_process_layer[i](quarter_feature)

            # 第一层的残差和融合
            original_combined = original_feature + self.down256_128[i](half_process) + original_process

            # 第二层的残差和融合
            half_combined = half_feature + self.up128_256[i](original_process) + self.down512_256[i](quarter_process) + half_process

            # 第三层的残差和融合
            quarter_combined = quarter_feature + self.up256_512[i](half_process) + quarter_process

            # 更新特征和处理层的输入
            original_feature = original_combined
            half_feature = half_combined
            quarter_feature = quarter_combined

        # print(original_combined.shape)
        # print(half_combined.shape)
        # print(quarter_combined.shape)
        final_emb = self.up128_512(original_combined) +  self.up256_512[-1](half_combined) + quarter_combined
        #print(final_emb.shape,"final_emb final_emb")
        feature = F.adaptive_avg_pool2d(final_emb, output_size=1).squeeze(-1).squeeze(-1)
        #print(feature.shape,"feature feature feature")
        return self.final_layer(final_emb),feature
# 创建模型实例
# model = MultiChannelModel()

# # 假设输入
# input_tensor = torch.randn(64, 5, 16, 193)  # (b, 5, 16, 193)

# # # 前向传播
# ans = model(input_tensor)


# print(f"ans: {ans.shape}")
# # 在终端中打印模型摘要
# summary(model, (5, 16, 193))
# # 创建 SummaryWriter 实例
# writer = SummaryWriter('runs/my_experiment')  # 使用不同的目录

# # 添加模型图
# writer.add_graph(model, input_tensor)

# # 执行其他 TensorBoard 日志记录操作...

# # 关闭 SummaryWriter
# writer.close()
# # print(f"Quarter Combined Shape: {quarter_combined.shape}")
