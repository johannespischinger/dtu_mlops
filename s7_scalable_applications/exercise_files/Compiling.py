import torch
import torch.nn as nn
import os
import time

from torchvision.datasets import CIFAR10
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timeit
from ptflops import get_model_complexity_info
import torch.utils.benchmark as benchmark

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()

        self.resnet = models.resnet152(pretrained=True, progress=True)
        self.fc = nn.Sequential(
            nn.Linear(1000, 10),
            nn.LogSoftmax(dim=1)

        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)

        return x

dataset = CIFAR10(root= os.getcwd(),train=False,transform=transforms.ToTensor(),download=True)
testloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = False)
resnet = Resnet()
scripted_resnet = torch.jit.script(Resnet())


def training(model, testloader):
    with torch.no_grad():
        test_acc = []
        start = time.time()
        for btx, (images, labels) in enumerate(testloader):
            prediction = torch.exp(model(images.float()))
            top_p, top_class = prediction.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_acc.append(torch.mean(equals.type(torch.FloatTensor)))

            if btx == 10:
                end = time.time()
                break

    time_res = end - start
    print(f"Accuracy: {sum(test_acc) / len(test_acc)}, Time: {time_res}")


t0 = benchmark.Timer(
    stmt='training(model,testloader)',
    setup='from __main__ import training',
    globals={'model': resnet,'testloader':testloader}
)

t1 = benchmark.Timer(
    stmt='training(model,testloader)',
    setup='from __main__ import training',
    globals={'model': scripted_resnet,'testloader':testloader}
)

print(t0.timeit(3))
print(t1.timeit(3))