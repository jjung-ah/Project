from torchvision.utils import save_image
from MyDataLoader import *
from torchvision.models.vgg import make_layers


import matplotlib.pyplot as plt
import numpy as np

#from vggModule import VGG
#from vggModule import *
from ResNetModule import ResNet
from ResNetModule import *
from configs import Configs
from train import train


def main():
    trainloader, testloader = make_train_val_set(Configs.data_root)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    cfg = {  # 8 + 3 =11 == vgg11
	    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	    # 10 + 3 = vgg 13
	    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	    # 13 + 3 = vgg 16
	    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	    # 16 +3 =vgg 19
	    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
	}
    conv = make_layers(cfg[Configs.model_config], batch_norm=True)
    #model = VGG(conv, num_classes=Configs.class_num, init_weights=True)
    #model = models.resnet18(pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50().to(device)
    #model = resnet18().to(device)
    #model = resnet152().to(device)   # 너무 커서 gpu 2개로도 out of memory 에러 발생
    print(model)

    train(model, trainloader, testloader)



if __name__ == '__main__':
	
    main()
