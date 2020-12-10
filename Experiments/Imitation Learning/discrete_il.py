# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:11:46 2020

@author: akpo2
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:16:00 2020

@author: akpo2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Discrete_IL_Net(nn.Module):
    def __init__(self,  input_shape, n_actions):
      super(Discrete_IL_Net, self).__init__()
      self.fc1 = nn.Linear(input_shape, 400)
      self.fc2 = nn.Linear(400,200)
      self.fc4 = nn.Linear(200,50)
      self.fc3 = nn.Linear(50,n_actions)
      self.dropout = nn.Dropout(p=0.5)
      self.softmax = nn.Softmax(dim=1)
    # x represents our data
    def forward(self, x):
      x = self.dropout(F.leaky_relu(self.fc1(x)))  
      x = self.dropout(F.leaky_relu(self.fc2(x)))
      x = self.dropout(F.leaky_relu(self.fc4(x)))
      x = self.softmax(self.fc3(x)) 
      return x