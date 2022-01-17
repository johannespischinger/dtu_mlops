import torch
import torch.nn as nn
import os
import time

from torchvision.datasets import CIFAR10
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ptflops import get_model_complexity_info


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet,self).__init__()
        
        self.resnet = models.resnet152(pretrained=True, progress = True)
        self.fc = nn.Sequential(
            nn.Linear(1000,10),
            nn.LogSoftmax(dim=1)

        )

    def forward(self,x):
        x = self.resnet(x)
        x = self.fc(x)

        return x


class Mobilenet(nn.Module):
    def __init__(self):
        super(Mobilenet,self).__init__()

        self.mobilenet = models.mobilenet_v3_large(pretrained=True, progress=True)
        self.fc = nn.Sequential(
            nn.Linear(1000, 10),
            nn.LogSoftmax(dim=1)

        )

    def forward(self, x):
        x = self.mobilenet(x)
        x = self.fc(x)

        return x


dataset = CIFAR10(root= os.getcwd(),train=False,transform=transforms.ToTensor(),download=True)
testloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = False)
resnet = Resnet()
mobilenet = Mobilenet()

with torch.no_grad():
    test_acc = []
    start = time.time()
    for btx,(images, labels) in enumerate(testloader):
        prediction = torch.exp(resnet(images.float()))
        top_p, top_class = prediction.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_acc.append(torch.mean(equals.type(torch.FloatTensor)))

        if btx == 10:
            end = time.time()
            break

    time_res = end-start
    macs, params = get_model_complexity_info(resnet, (3, 32, 32), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print(f"Accuracy: {sum(test_acc) / len(test_acc)}, Time: {time_res}")

with torch.no_grad():
    test_acc = []
    start = time.time()
    for btx,(images, labels) in enumerate(testloader):
        prediction = torch.exp(mobilenet(images.float()))
        top_p, top_class = prediction.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_acc.append(torch.mean(equals.type(torch.FloatTensor)))

        if btx == 10:
            end = time.time()
            break

    time_mobile = end-start
    macs, params = get_model_complexity_info(mobilenet, (3, 32, 32), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print(f"Accuracy: {sum(test_acc) / len(test_acc)}, Time: {time_mobile}")