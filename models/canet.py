# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 15:41:25 2021

@author: axmao2-c
"""

import torch.nn as nn
import torch


class CaNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        
        self.conv1 = BasicBlock_b(in_channels=1, out_channels=16, kernel_size=(3,3))
        self.conv2 = BasicBlock_b(in_channels=16, out_channels=16, kernel_size=(3,3))
        self.conv3 = BasicBlock_b(in_channels=16, out_channels=32, kernel_size=(3,3))
        self.conv4 = BasicBlock_b(in_channels=32, out_channels=32, kernel_size=(3,3))
        self.conv5 = BasicBlock_b(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.conv6 = BasicBlock_b(in_channels=64, out_channels=64, kernel_size=(3,3))
        
        self.maxpool = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1),padding=(1,0))
        #self.attention_channel = SELayer(64,4)
        #self.attention_map = Feature_map_att(1,16)
        # self.conv7 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=(1,1)),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))

        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_maxpool = nn.MaxPool2d(kernel_size=(11,1),stride=(11,1),padding=(0,0))
        
        self.lstm = nn.LSTM(input_size=64,hidden_size=128,batch_first=True)
        self.fc = nn.Linear(128, num_classes)
        self.sigma = nn.Sigmoid()
        
    # def _init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             # m.weight.data.normal_(0, math.sqrt(2. / n))
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.maxpool(output)
        print(output.size())
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.maxpool(output)
        print(output.size())
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.maxpool(output)
        print(output.size())
        #output = self.attention_channel(output)
        #output, att_map = self.attention_map(output)
        # output = self.conv7(output)
        # output = self.avg_pool(output) #[batch_size, num_filters, 1, 1]
        output = self.global_maxpool(output)
        print(output.size())
        # print(output.size())
        
        output = output.view(output.size(0),output.size(3),output.size(1)) #[batch_size, num_filters]
        output, (h_n, h_c) = self.lstm(output)
        print(output[:,-1,:].size())
        output = self.fc(output[:,-1,:])
        output = self.sigma(output)
        # print(output.size())
    
        return output

class BasicBlock_b(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3)):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(out_channels)
        )
        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1)),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.residual_function(x)
        output = nn.ReLU(inplace=True)(out + self.shortcut(x))
        
        return output

#Spatial-wise
class Feature_map_att(nn.Module):
    def __init__(self, input_channel=1, middle_channel=16): #dim_acc, dim_gyr= 64
        super().__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, middle_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(
            nn.Conv2d(middle_channel, input_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(input_channel),
            nn.Sigmoid())
        
    def forward(self, features):
        squeezed_features = torch.mean(features, dim=1, keepdim=True, out=None)
        excitation = self.conv_1(squeezed_features)
        out = self.conv_2(excitation)
      
        return features * out.expand_as(features), out

#channel-wise
class SELayer(nn.Module):
    def __init__(self, channel = 64, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
            )
        self.fc_2 = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
            )

    def forward(self, features):
        b, c, _, _ = features.size()
        squeeze = self.avg_pool(features).view(b, c)
        excitation = self.fc_1(squeeze)
        acc_out = self.fc_2(excitation).view(b, c, 1, 1)
        
        return features * acc_out.expand_as(features)


def canet():

    return CaNet()

net = canet()
print(net)
data = torch.Tensor(4, 1, 88, 87)
print(net(data))

# import torch
# from torch.autograd import Variable
# import torch.nn.functional as F
# import torch.nn as nn

# sample = Variable(torch.ones(2,2))
# a = torch.Tensor(2,2)
# a[0,0] = 0
# a[0,1] = 1
# a[1,1] = 2
# a[1,1] = 3
# target = Variable (a)



