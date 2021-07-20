import torch.optim as optim
import time
import cv2
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
#import torchvision.transforms as transforms
from skimage import io, transform
from torchvision import transforms
import torchvision.models as models

import torch.nn as nn
import torch.nn.functional as F

classes = ['fear', 'happy', 'sad', 'neutral', 'disgust', 'surprise', 'angry']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.densenet161(pretrained=True)
model.load_state_dict(torch.load("/content/drive/Shareddrives/zeogi_gogi/densenet_83_model.pth"))
#print(model)
''' 모델의 out feature가 class 1000이라 7로 바꾸어 주려 했으나 잘 되지 않음..
list(model.children())[1].out_feature = 7
model = model
model_children = list(model.children())
#print(model_children[1])
result = model_children[1]
print(result.out_features)
result.out_features = 7
print(model_children[1])
#model = torch.load("/content/drive/Shareddrives/zeogi_gogi/models/jjung_model/vgg16_epoch100_accuracy92.875.pt")
#model = torch.load("/content/drive/Shareddrives/zeogi_gogi/densenet_83_model.pth")
#for param in model.classifier.parameters(): 
#    param.requires_grad = False
'''
model.eval()



def test(path):
    input_image = Image.open(path)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    #input_tensor = input_image
    #input_tensor = torch.from_numpy(input_tensor).float()
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    print(input_tensor.shape)
    print(input_batch.shape)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        print(output.shape)
        print(output[0].shape)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    #print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities.shape)


    #single_prediction = model(output)
    print('Prediction: ', torch.argmax(probabilities).item(), classes[torch.argmax(probabilities).item()])

    plt.imshow(input_image)
    img = plt.show()
    
    return torch.argmax(probabilities).item(), classes[torch.argmax(probabilities).item()]