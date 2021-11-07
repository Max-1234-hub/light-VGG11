import torch.nn as nn
import torch

#channel-wise
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

#axis-wise
class ASELayer(nn.Module):
    def __init__(self, channel, increment_channel=64):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, increment_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(increment_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0,2,1,3)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        return out.permute(0,2,1,3)

#cross attention by channel-wise
class CA_channel(nn.Module):
    def __init__(self, dim_acc, dim_gyr, ratio): #dim_acc, dim_gyr= 64
        super().__init__()
        dim = dim_acc + dim_gyr #128
        dim_out = int(2*dim/ratio) #ratio=16
        self.fc_squeeze = nn.Linear(dim, dim_out)
      
        self.fc_acc = nn.Linear(dim_out, dim_acc)
        self.fc_gyr = nn.Linear(dim_out, dim_gyr)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_acc, f_gyr):
        b, c, _, _ = f_acc.size()
        squeeze_array = []
        for tensor in [f_acc, f_gyr]:
          tview = tensor.view(tensor.shape[:2] + (-1,)) #[256,64,600]|3*200
          squeeze_array.append(torch.mean(tview, dim=-1)) #[256,64]
        squeeze = torch.cat(squeeze_array, 1)
      
        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)
      
        acc_out = self.fc_acc(excitation)
        gyr_out = self.fc_gyr(excitation)
      
        acc_out = self.sigmoid(acc_out).view(b, c, 1, 1)
        gyr_out = self.sigmoid(gyr_out).view(b, c, 1, 1)
      
        return f_acc * acc_out.expand_as(f_acc), f_gyr * gyr_out.expand_as(f_gyr)


    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=(1,stride), padding=(0,1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1), bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.CA_C = CA_channel(64,64,16)

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1,stride), bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x_acc, x_gyr):
        out_acc = self.residual_function(x_acc)
        out_gyr = self.residual_function(x_gyr)
        
        output_acc, output_gyr = self.CA_C(out_acc, out_gyr)
        acc_output = nn.ReLU(inplace=True)(output_acc + self.shortcut(x_acc))
        gyr_output = nn.ReLU(inplace=True)(output_gyr + self.shortcut(x_gyr))
        
        return acc_output, gyr_output


class ResNet(nn.Module):

    def __init__(self, num_classes=6):
        super().__init__()

        self.conv1_acc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1,3), padding=(0,1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.conv1_gyr = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1,3), padding=(0,1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))
        self.conv2_x = BasicBlock(in_channels=32, out_channels=64,stride=1)
        # self.conv2_gyr = BasicBlock(in_channels=32, out_channels=64,stride=1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        xa = x[:,:,:,0:3]
        xg = x[:,:,:,3:6]
        
        xa = xa.permute(0,1,3,2)
        output_x = self.conv1_acc(xa)
        output_x = self.maxpool(output_x)
        xg = xg.permute(0,1,3,2)
        output_y = self.conv1_gyr(xg)
        output_y = self.maxpool(output_y)
        
        output_x,output_y = self.conv2_x(output_x, output_y)
        # print(output_x.size())
       
        output_x = self.avg_pool(output_x) #[batch_size, num_filters, 1, 1]
        output_x = output_x.view(output_x.size(0), -1)
        output_y = self.avg_pool(output_y)
        output_y = output_y.view(output_y.size(0), -1)

        output = torch.cat((output_x,output_y), 1) #[batch_size, 2*num_filters]
        output = self.fc(output)

        return output

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet()

# net = resnet18()
# print(net)
# data = torch.Tensor(4,1,200,6)
# print(net(data))


