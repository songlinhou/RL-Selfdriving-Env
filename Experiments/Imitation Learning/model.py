# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:16:00 2020

@author: akpo2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,  input_shape, n_actions):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(input_shape, 32)
      self.fc2 = nn.Linear(32,16)
      self.fc3 = nn.Linear(16,n_actions)
      self.dropout = nn.Dropout(p=0.2)
    # x represents our data
    def forward(self, x):
      x = self.dropout(F.leaky_relu(self.fc1(x)))  
      x = self.dropout(F.leaky_relu(self.fc2(x)))
      x = F.leaky_relu(self.fc3(x)) 
      return x