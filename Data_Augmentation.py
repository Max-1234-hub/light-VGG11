# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:04:50 2021

@author: axmao2-c
"""

#Data augmentation
import torch
import numpy as np
import numpy
import skimage
import random

#New Image Same Class
def new_image_same_class(spectrogram, data_same_class):
    """
    Generate new image by adding spectrogram from the same class
    """
    index = np.random.randint(0, len(data_same_class))
    alpha = random.uniform(0,1)
    new_spectrogram = alpha*spectrogram + (1-alpha)*data_same_class[index]

    return new_spectrogram

#Add Gaussian Noise
def add_gaussian_noise(spectrogram):
    """
    Add noise
    """
    noise_gs_spectrogram = skimage.util.random_noise(spectrogram,mode ='gaussian',
                                   seed=None,clip=True,mean=0,var=0.03)
    
    return torch.from_numpy(noise_gs_spectrogram).float()


