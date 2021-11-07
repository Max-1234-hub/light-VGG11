# Implementation of SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
# Ref: https://arxiv.org/pdf/1904.08779.pdf

import random
import numpy as np
#import tensorflow as tf
#from tensorflow_addons.image import sparse_image_warp


class SpecAugment():
    '''
    Augmentation Parameters for policies
    -----------------------------------------
    Policy | W  | F  | m_F |  T  |  p  | m_T
    -----------------------------------------
    None   |  0 |  0 |  -  |  0  |  -  |  -
    -----------------------------------------
    LB     | 80 | 27 |  1  | 100 | 1.0 | 1
    -----------------------------------------
    LD     | 80 | 27 |  2  | 100 | 1.0 | 2
    -----------------------------------------
    SM     | 40 | 15 |  2  |  70 | 0.2 | 2
    -----------------------------------------
    SS     | 40 | 27 |  2  |  70 | 0.2 | 2
    -----------------------------------------
    
    LB  : LibriSpeech basic
    LD  : LibriSpeech double
    SM  : Switchboard mild
    SS  : Switchboard strong
    W   : Time Warp parameter
    F   : Frequency Mask parameter
    m_F : Number of Frequency masks
    T   : Time Mask parameter
    p   : Parameter for calculating upper bound for time mask
    m_T : Number of time masks
    '''
    
    def __init__(self, mel_spectrogram, policy, zero_mean_normalized=True):
        self.mel_spectrogram = mel_spectrogram
        self.policy = policy
        self.zero_mean_normalized = zero_mean_normalized
        
        # Policy Specific Parameters
        if self.policy == 'LB':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 1, 100, 1.0, 1
        elif self.policy == 'LD':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 2, 100, 1.0, 2
        elif self.policy == 'SM':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 15, 2, 70, 0.2, 2
        elif self.policy == 'SS':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 27, 2, 70, 0.2, 2
        elif self.policy == 'max':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 5, 15, 3, 5, 0.2, 2
        
    
    def freq_mask(self):
        
        v = self.mel_spectrogram.size()[1] # no. of mel bins
        # self.mel_spectrogram = np.reshape(self.mel_spectrogram, (-1, self.mel_spectrogram.shape[0], self.mel_spectrogram.shape[1], 1))
        
        # apply m_F frequency masks to the mel spectrogram
        for i in range(self.m_F):
            f = int(np.random.uniform(0, self.F)) # [0, F)
            f0 = random.randint(0, v - f) # [0, v - f)
            self.mel_spectrogram[:, f0:f0 + f, :] = 0
            
        return self.mel_spectrogram
    
    
    def time_mask(self):
    
        tau = self.mel_spectrogram.size()[2] # time frames
        # self.mel_spectrogram = np.reshape(self.mel_spectrogram, (-1, self.mel_spectrogram.shape[0], self.mel_spectrogram.shape[1], 1))
        
        # apply m_T time masks to the mel spectrogram
        for i in range(self.m_T):
            t = int(np.random.uniform(0, int(tau*self.p))) # [0, T)
            t0 = random.randint(0, tau - t) # [0, tau - t)
            self.mel_spectrogram[:, :, t0:t0 + t] = 0
            
        return self.mel_spectrogram
