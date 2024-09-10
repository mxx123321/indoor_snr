import torch.nn as nn
import torch
import numpy as np
import torch
from einops import repeat
from torch import nn, Tensor
from torch.nn import functional as F
from torch import nn, Tensor
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)
#--mask_block_size 64 --mask_ratio 0.7
class Masking(nn.Module):
    def __init__(self, block_size, ratio):
        super(Masking, self).__init__()

        self.block_size = block_size
        self.ratio = ratio


    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()
        
        
        #torch.cat((a,b,c),axis = 0)
        
        #print(input_mask.shape,"input_mask input_mask input") # [64, 1, 16, 193]
        input_mask = resize(input_mask, size=(H, W))
        #print(input_mask.shape,"input_mask input_mask after") # [64, 1, 16, 193]
        #print(img.shape,       "img img after")               # [64, 5, 16, 193]
        masked_img = img * input_mask

        return masked_img
    
    
    
class MultiMasking(nn.Module):
    def __init__(self, block_size, ratio):
        super(MultiMasking, self).__init__()

        self.block_size = block_size
        self.ratio = ratio


    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()
        
        input_mask_2 = torch.rand(mshape, device=img.device)
        input_mask_2 = (input_mask_2 > self.ratio).float()
        
        input_mask_3 = torch.rand(mshape, device=img.device)
        input_mask_3 = (input_mask_3 > self.ratio).float()
        
        input_mask_4 = torch.rand(mshape, device=img.device)
        input_mask_4 = (input_mask_4 > self.ratio).float()
        
        input_mask_5 = torch.rand(mshape, device=img.device)
        input_mask_5 = (input_mask_5 > self.ratio).float()
        #torch.cat((a,b,c),axis = 0)
        input_mask = torch.cat((input_mask,input_mask_2,input_mask_3,input_mask_4,input_mask_5),axis = 1)
        #print(input_mask.shape,"input_mask input_mask input") # [64, 1, 16, 193]
        input_mask = resize(input_mask, size=(H, W))
        #print(input_mask.shape,"input_mask input_mask after") # [64, 1, 16, 193]
        #print(img.shape,       "img img after")               # [64, 5, 16, 193]
        masked_img = img * input_mask
        #print("MultiMasking")

        return masked_img