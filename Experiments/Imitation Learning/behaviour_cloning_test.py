# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 18:39:42 2020

@author: akpo2
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from model import Net
from discrete_il import Discrete_IL_Net

data_dir = 'D:\RL_project\FInal Project\RLCar\Build'
#columns = ['SD0','SD1','SD2','SD3','SD4','SD5','SD6','RotY','RotW','DistRatio','steer','acceleration']
# data = pd.read_csv(os.path.join(data_dir,'record_data.csv'), names=columns)

data = pd.read_csv(os.path.join(data_dir,'record_data.csv'))
data.head()
data = data.astype(float)

num_bins = 25
samples_per_bin = 150
hist_steer, bins_steer = np.histogram(data['steer'],num_bins)
print(bins_steer)
center = (bins_steer[:-1]+bins_steer[1:])*0.5
plt.bar(center,hist_steer,width=0.05)

print('total data',len(data))
remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steer'])):
        if data['steer'][i] >= bins_steer[j] and data['steer'][i] <= bins_steer[j+1]:
            list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)
    
print('removed', len(remove_list))
data.drop(data.index[remove_list], inplace = True)
print('after removing data',len(data))
hist, bins = np.histogram(data['steer'],num_bins)
plt.bar(center,hist,width=0.05)

num_of_input_states = 18
index_value = num_of_input_states
print(data.iloc[1])        
def load_data(data_dir,df):
    sensor_data =[]
    steer = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
#        print('i ',i)
        sensor_data.append(indexed_data[:index_value])
        steer.append(indexed_data[index_value+1:])
        
#    sensor_data = np.asarray(sensor_data)
#    steer = np.asarray(steer)
    return sensor_data,steer
        
sensor_data,steer_data = load_data(data_dir,data)
X_train, X_valid, y_train, y_valid = train_test_split(sensor_data,steer_data,test_size=0.2, random_state= 6)                                   
print('Training Samples: {}\nValid Sample: {}'.format(len(X_train), len(X_valid)))


#fig, axes = plt.subplots(1,2, figsize=(12,4))
#axes[0].hist(y_train, bins=num_bins, width = 0.05)
#axes[0].set_title('Training set')      
#axes[1].hist(y_valid, bins=num_bins, width = 0.05)
#axes[1].set_title('Validation set')  


#X_train = torch.from_numpy(X_train.to_numpy()).float()
#y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())
#X_valid = torch.from_numpy(X_valid.to_numpy()).float()
#y_valid = torch.squeeze(torch.from_numpy(y_valid.to_numpy()).float())

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
#y_train = torch.LongTensor(y_train)
y_train = y_train - 1
X_valid = torch.FloatTensor(X_valid)
y_valid = torch.FloatTensor(y_valid)
#y_valid = torch.LongTensor(y_valid)
y_valid = y_valid - 1

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")
X_train = X_train.to(device)
y_train = y_train.to(device)
X_valid = X_valid.to(device)
y_valid = y_valid.to(device)

#n_actions = 25
n_actions = 2
model_req = Net(X_train.shape[1],n_actions).to(device)
#model_req=Discrete_IL_Net(X_train.shape[1],n_actions).to(device)
print(model_req)
optimiser = optim.Adam(model_req.parameters(), lr =0.001)
#optimiser = optim.Adam(model_req.parameters(), lr =0.01)
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

train_loss=[]
val_loss=[]
accuracy_list = []
epochs_tot = 1000000
least_loss = None
meanScore = 0
for e in range(epochs_tot):
    output = model_req(X_train).to(device)
#    loss = criterion(output,y_train.squeeze())
    loss = criterion(output,y_train)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    train_loss.append(loss.item())
    print('Epoch : ',e,' loss : ',loss.item())
    if e % 5 == 0:
        with torch.no_grad():
            eval_output = model_req(X_valid).to(device)
#            eval_loss = criterion(eval_output, y_valid.squeeze())
            eval_loss = criterion(eval_output,y_valid)
            validation_loss = eval_loss.item()
#            accuracy = torch.sum(output.argmax(axis=1) == y_train.squeeze()) / output.shape[0]
            print("=============== Validation loss after 100 epoch : =================",eval_loss.item())
#            print("=============== Accuracy after 100 epoch : =================",accuracy)
            val_loss.append(eval_loss.item())
#            accuracy_list.append(accuracy)
            if least_loss is None or validation_loss < least_loss:
                        torch.save(model_req.state_dict(),f'D:\RL_project\FInal Project\RLCar\Path_folder\{e}_{validation_loss}.pth')
                        if least_loss is not None:
                            print("Best mean reward updated %.3f -> %.3f, model saved" % (least_loss, validation_loss))
                        least_loss = validation_loss
                    
        
   
plt.plot(train_loss)
plt.plot(val_loss)
plt.plot(accuracy_list)
plt.legend(['training','valiation'])
plt.title('Loss')
plt.xlabel('Epoch')

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

def get_il_state(state):
    #idx = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,30,31,32,33]
    idx = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,30,31,32]
    return state[idx]

state = torch.load('D:/RL_project/FInal Project/RLCar/Path_folder/46305_0.172707200050354.pth')
def get_action(state):
    if len(state) == 34:
        state = get_il_state(state)
    with torch.no_grad():
        state = torch.Tensor(state).view(1,-1).to(device)
        print("state.shape=",state.shape)
        action = model_req(state)
    return action.cpu().numpy()

def il_eval():
    state = env.reset()
    score = 0
    max_t = 10000
    for t in range(max_t):
        action = get_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = get_il_state(next_state)
        state = next_state
        score += reward
        if done:
            break 
        
#env = UnityToGymWrapper(UnityEnvironment(base_port=5004), 0)
env = UnityToGymWrapper(UnityEnvironment('D:/RL_project/FInal Project/RLCar/Build/RLCar.exe'), 0)

il_eval()

env.close()