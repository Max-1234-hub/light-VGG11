""" train and test dataset

author baiyu
"""

import torch
import random
from augment import SpecAugment
from torch.utils.data import Dataset
from Data_Augmentation import   new_image_same_class, add_gaussian_noise


class My_Dataset(Dataset):

    def __init__(self, pathway, data_id, transform_auto=None):
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test = torch.load(pathway)
        if data_id == 0:
            self.data, self.labels = X_train, Y_train
            self.transform = True
        elif data_id == 1:
            self.data, self.labels = X_valid, Y_valid
            self.transform = False
        elif data_id == 2:
            self.data, self.labels = X_test, Y_test
            self.transform = False
        
        self.transform_auto = transform_auto
       
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.labels[index] #torch.size: [1]
        image = self.data[index] #troch.size: [1,128,44]
        index_same_class = torch.where(self.labels == label)
        data_same_class = self.data[index_same_class]
        # if self.transform:
            # if random.uniform(0,1) < 0.5: 
            #     apply = SpecAugment(image, 'max')
            #     image = apply.time_mask()
            # if random.uniform(0,1) < 0.5: 
            #     apply = SpecAugment(image, 'max')
            #     image = apply.freq_mask()
            # if random.uniform(0,1) < 0.5:
            #     image = new_image_same_class(image, data_same_class)
            #if random.uniform(0,1) < 0.5:
             #   image = add_gaussian_noise(image)
                
        return image, label