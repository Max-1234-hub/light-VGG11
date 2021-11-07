""" helper function

author baiyu
"""

import sys
import torch
from torchvision import transforms
# from Data_Augmentation import time_shift_spectrogram,  pitch_shift_spectrogram

import numpy as np
from torch.utils.data.sampler import  WeightedRandomSampler

from torch.utils.data import DataLoader

from dataset import My_Dataset

def get_network(args):
    """ return given network
    """
    
    #Bird voice recognition
    if args.net == 'bv_cnn':
        from models.bv_net import bv_cnn
        net = bv_cnn()
    elif args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'canet':
        from models.canet import canet
        net = canet()
    elif args.net == 'resnet18':
        from models.resnet_raw import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet_raw import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet_raw import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet_raw import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet_raw import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

def get_os_dataloader(pathway, data_id = 0, batch_size=16, num_workers=2, shuffle=True):
    Mydataset = My_Dataset(pathway, data_id, transform=None)
    all_labels = [label for data, label in Mydataset]
    number = np.unique(all_labels, return_counts = True)[1]
    weight = 1./ torch.from_numpy(number).float()
    weights = [weight[int(i)] for i in all_labels]
    
    sampler = WeightedRandomSampler(weights, num_samples=len(Mydataset), replacement=True)
    Data_loader = DataLoader(Mydataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, sampler=sampler)
    
    return Data_loader

def get_mydataloader(pathway, data_id = 1, batch_size=16, num_workers=2, shuffle=True):
    
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop([88,87], padding=0),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    Mydataset = My_Dataset(pathway, data_id, transform_auto = transform_train)
    Data_loader = DataLoader(Mydataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    return Data_loader

# def get_testdata(pathway):
#     X_train, X_valid, X_test, Y_train, Y_valid, Y_test = torch.load(pathway)
    
#     return X_test, Y_test