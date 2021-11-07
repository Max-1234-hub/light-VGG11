# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:27:38 2021

@author: axmao2-c
"""

import torch.nn as nn
import torch


class Bv_Cnn(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=11),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=15),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=15),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        
        self.maxpool = nn.MaxPool1d(kernel_size=7)
        # self.dropout = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(384,256)  
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(384,num_classes)      

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        # print(out.size())
        out = self.conv2(out)
        out = self.maxpool(out)
        # print(out.size())
        out = self.conv3(out)
        out = self.maxpool(out)
        # print(out.size())
        out = self.conv4(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        # print(out.size())
        
        out = out.view(out.size(0),-1)
        output = self.fc2(out)

        return output

def bv_cnn():

    return Bv_Cnn()

net = bv_cnn()
print(net)
data = torch.Tensor(4,1,22050)
print(net(data))