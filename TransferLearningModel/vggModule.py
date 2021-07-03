import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from configs import Configs


class VGG(nn.Module):

    def __init__(self, conv, num_classes=100, init_weights=True):
        super(VGG, self).__init__()  # pytorch에서 class 형태의 모델은 항상 nn.Module을 상속받아야 하며, 
                                     # super(모델명, self).init()을 통해 nn.Module.init()을 실행시키는 코드가 필요

       
    return nn.Sequential(*layers)
