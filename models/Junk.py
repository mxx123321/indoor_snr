import torch
import torch.nn as nn
import torch.nn.functional as F
#import onnx
#import onnx.utils
#import onnx.version_converter
#from  tensorboardX import  SummaryWriter

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class PreInput(nn.Module):
    def __init__(self):
        super(PreInput, self).__init__()
        self.input_channels = 5
        self.out_channels = 256
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels, self.out_channels, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) #正好维度减半
        )
    def forward(self,x):
        return self.conv(x)

class Layer1(nn.Module):
    def __init__(self):
        super(Layer1, self).__init__()
        self.input_channels = 256
        self.conv = nn.Sequential(
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(),
            nn.Conv2d(self.input_channels,64,(1,1),1),
        )
    def forward(self,x):
        return self.conv(x)

class Layer2(nn.Module):
    def __init__(self):
        super(Layer2, self).__init__()
        self.input_channels = 256
        self.conv = nn.Sequential(
           nn.BatchNorm2d(self.input_channels),
           nn.ReLU(),
           nn.Conv2d(self.input_channels,self.input_channels,(5,1),1,padding='same'),
           nn.Conv2d(self.input_channels,32,(1,3),1,padding='same')
        )
    def forward(self,x):
        return self.conv(x)

class Layer3(nn.Module):
    def __init__(self):
        super(Layer3, self).__init__()
        self.input_channels = 256
        self.conv = nn.Sequential(
           nn.AdaptiveMaxPool2d((8,96)),

           nn.BatchNorm2d(self.input_channels),
           nn.ReLU(),
           nn.Conv2d(self.input_channels,32,(3,3),padding='same'),
        )
    def forward(self,x):
        return self.conv(x)

class Layer4(nn.Module):
    def __init__(self):
        super(Layer4, self).__init__()
        self.input_channels = 256
        self.conv = nn.Sequential(
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(),
            nn.Conv2d(self.input_channels,32,(1,1),1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 128, (1,1), 1),
        )
    def forward(self,x):
        return self.conv(x)

class Upper(nn.Module):
    def __init__(self):
        super(Upper, self).__init__()
        self.layer1 = Layer1()
        self.layer2 = Layer2()
        self.layer3 = Layer3()
        self.layer4 = Layer4()
    def forward(self,x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x)
        # 合成256
        res = torch.cat([x1,x2,x3,x4],dim=1)
        # 残差
        return res + x

class Lower(nn.Module):
    def __init__(self):
        super(Lower, self).__init__()
        self.UpperMaxPool1 = nn.AdaptiveMaxPool2d((4,48))
        self.UpperConv1 = LowerSub1()
        self.DownMaxPool1 = nn.AdaptiveMaxPool2d((4,48))
        self.DownConv1 =  LowerSub2()
        ##########
        self.UpperMaxPool2 = nn.AdaptiveMaxPool2d((2,24))
        self.UpperConv2 = LowerSub3()
        self.DownMaxPool2 = nn.AdaptiveMaxPool2d((2,24))
        self.DownConv2 =  LowerSub4()
        ##############
        self.UpperMaxPool3 = nn.AdaptiveMaxPool2d((1,12))
        self.UpperConv3 = LowerSub5()
        self.DownMaxPool3 = nn.AdaptiveMaxPool2d((1,12))
        self.DownConv3 =  LowerSub6()
    def forward(self,x):
        upper = self.UpperMaxPool1(x)
        down = self.DownMaxPool1(x)

        ori_upper = upper
        ori_down = down

        upper = self.UpperConv1(upper)
        res_upper = upper
        down = self.DownConv1(upper)
        res_down = down

        upper = ori_upper + res_upper + res_down
        down = ori_down + res_upper + res_down
        ################
        upper = self.UpperMaxPool2(upper)
        down = self.DownMaxPool2(down)

        ori_upper = upper
        ori_down = down

        upper = self.UpperConv2(upper)
        res_upper = upper
        down = self.DownConv2(upper)
        res_down = down

        upper = ori_upper + res_upper + res_down
        down = ori_down + res_upper + res_down
        ########################
        upper = self.UpperMaxPool3(upper)
        down = self.DownMaxPool3(down)

        ori_upper = upper
        ori_down = down

        upper = self.UpperConv3(upper)
        res_upper = upper
        down = self.DownConv3(upper)
        res_down = down

        upper = ori_upper + res_upper + res_down
        down = ori_down + res_upper + res_down
        return upper + down

#上1
class LowerSub1(nn.Module):
    def __init__(self):
        super(LowerSub1, self).__init__()
        self.dim = 256
        self.conv = nn.Sequential(
            nn.BatchNorm2d(self.dim),
            nn.ReLU(),
            nn.Conv2d(self.dim,self.dim,(3,3),padding='same'),
        )
    def forward(self,x):
        return self.conv(x)

#下1
class LowerSub2(nn.Module):
    def __init__(self):
        super(LowerSub2, self).__init__()
        self.dim = 256
        self.conv = nn.Sequential(
            nn.BatchNorm2d(self.dim),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, (1,1), padding='same'),
        )
    def forward(self,x):
        return self.conv(x)

#上2
class LowerSub3(nn.Module):
    def __init__(self):
        super(LowerSub3, self).__init__()
        self.dim = 256
        self.conv = nn.Sequential(
            nn.BatchNorm2d(self.dim),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, (3,3), padding='same'),
        )
    def forward(self,x):
        return self.conv(x)

#下2
class LowerSub4(nn.Module):
    def __init__(self):
        super(LowerSub4, self).__init__()
        self.dim = 256
        self.conv = nn.Sequential(
            nn.BatchNorm2d(self.dim),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, (1,1), padding='same'),
        )
    def forward(self,x):
        return self.conv(x)

#上3
class LowerSub5(nn.Module):
    def __init__(self):
        super(LowerSub5, self).__init__()
        self.dim = 256
        self.conv = nn.Sequential(
            nn.BatchNorm2d(self.dim),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, (3,3), padding='same'),
        )
    def forward(self,x):
        return self.conv(x)

#下3
class LowerSub6(nn.Module):
    def __init__(self):
        super(LowerSub6, self).__init__()
        self.dim = 256
        self.conv = nn.Sequential(
            nn.BatchNorm2d(self.dim),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, (1,1), padding='same'),
        )
    def forward(self,x):
        return self.conv(x)

class EndLayer(nn.Module):
    def __init__(self):
        super(EndLayer, self).__init__()
        self.input_channels = 256
        self.output_channels = 256
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels,self.output_channels ,(1,1),1),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 6)),
            nn.Flatten(),
            nn.Linear(1536,3),
        )
    def forward(self, x):
        return self.conv(x)


class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.pre = PreInput()
        self.upper = Upper()
        self.lower = Lower()
        self.end = EndLayer()
    def forward(self,x):
        x = self.pre(x)
        #print(x.shape)
        x = self.upper(x)
        #print(x.shape)
        x = self.lower(x)
        #print(x.shape)
        temp = F.adaptive_avg_pool2d(x, output_size=1).squeeze(-1).squeeze(-1)
        print(temp.shape,"temp temp")
        x = self.end(x)
        return x, temp



# if __name__ == '__main__':
#     input = torch.rand(16,5,16,193)
#     # pre_layer = PreInput()
#     # print(pre_layer(input).shape)
#     #
#     # input = pre_layer(input)
#     # layer1 = Layer1()
#     # layer2 = Layer2()
#     # layer3 = Layer3()
#     # layer4 = Layer4()
#     # print(layer1(input).shape)
#     # print(layer2(input).shape)
#     # print(layer3(input).shape)
#     # print(layer4(input).shape)
#     #
#     # upper = Upper()
#     # print(upper(input).shape)
#     # tmp = upper(input)
#     #
#     # sub_layer1 = Lower()
#     # print(sub_layer1(tmp).shape)
#     # tmp = sub_layer1(tmp)
#     #
#     # end_layer = EndLayer()
#     # print(end_layer(tmp).shape)
#     model = FullModel()
#     out = model(input)
#     with SummaryWriter(comment='gkx')as w:
#         w.add_graph(model, (input,))
    # from torchviz import make_dot
    #
    # MyConvNetVis = make_dot(out, params=dict(list(model.named_parameters()) + [('x', input)]))
    # MyConvNetVis.format = "png"
    # # 指定文件生成的文件夹
    # MyConvNetVis.directory = "data"
    # # 生成文件
    # MyConvNetVis.view()