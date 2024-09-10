import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict



class TestPos(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv1 = nn.Conv2d(5,5,1,1,0)

        self.linear = nn.Linear(5,3)
    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.conv1.parameters(), "lr": 0.1*base_lr },
            {"params": self.linear.parameters(), "lr": 1.0 * base_lr}
            
        ]

        return params

    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = F.adaptive_avg_pool2d(x1, 1).squeeze(-1).squeeze(-1)

        x2 = self.linear(x1)

        return x2,x1
    

