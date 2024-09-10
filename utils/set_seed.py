import torch
import random
import numpy as np
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证在有多个GPU时也有可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

