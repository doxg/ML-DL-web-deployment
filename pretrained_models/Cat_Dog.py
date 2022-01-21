import csv
import pdb
import torch as t
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import shutil
import random
import time
from PIL import Image
import numpy as np


class Model(t.nn.Module):
    def __init__(self, ResnetBlock, train_on):
        super(Model, self).__init__()
        self.in_channels = 64
        self.initial1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.layer = self.make_res_layer(ResnetBlock, 1, intermediate_channels=64, stride=1)

        self.fc1 = t.nn.Linear(32768, 1024)
        self.fc2 = t.nn.Linear(1024, 512)
        self.fc3 = t.nn.Linear(512, 2)

        self.dropout_use = train_on
        self.dropout = t.nn.Dropout(p=0.25)

        self.normalize1 = nn.BatchNorm1d(1024)
        self.normalize2 = nn.BatchNorm1d(512)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x):
        init_size = x.size(0)
        #print(init_size)

        output1 = self.initial1(x)
        output1 = self.bn1(output1)
        output1 = self.relu(output1)
        output1 = self.maxpool(output1)

        block1 = self.layer(output1)
        block1 = self.avgpool(block1)
       # pdb.set_trace()
        block2 = self.layer(output1)
        block2 = self.avgpool(block2)

        output = [block1, block2]
        output = t.cat(output, 1)
        output = output.view(init_size, -1)

       # pdb.set_trace()
        output5 = self.normalize1(self.fc1(output))
        output5 = F.relu(output5)

        if self.dropout_use:
            output5 = self.normalize2(self.fc2(output5))
            output5 = self.dropout(F.relu(output5))  # Applied Dropout
            output5 = F.relu(self.fc3(output5))

        else:
            output5 = self.normalize2(self.fc2(output5))
            output5 = F.relu(self.fc3(output5))

        # pdb.set_trace()

        return F.softmax(output5, dim=1)

    def make_res_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        layers = []
        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead

        layers.append(
            block(self.in_channels, intermediate_channels, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 2

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)



class ResnetBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, stride=1):
        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)

        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)

        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(intermediate_channels)

        self.resize = nn.Sequential(nn.Conv2d(in_channels, intermediate_channels*4, kernel_size=1, stride=stride), nn.BatchNorm2d(intermediate_channels*4))
        self.relu = nn.ReLU()

    def forward(self, x):
        init_size = x.size(0)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        #pdb.set_trace()
        if identity.shape[1] != x.shape[1]:
            identity = self.resize(identity)


        x += identity
        x = self.relu(x)

        return x


if __name__ == "__main__":
    model = Model(ResnetBlock, train_on=False)
    model.cuda()
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)





