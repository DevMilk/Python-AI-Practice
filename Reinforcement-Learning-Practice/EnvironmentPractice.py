# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 22:46:21 2020

@author: Ugur
"""

import gym, random
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np
from collections import deque
 
env = gym.make("Taxi-v3").env
env.render()

env.reset()

#%%

print("State space: ", env.observation_space)
print("Action space: ", env.action_space)

#Bu kombinasyonun hangi stateye karşılık geldiğini gösterir
state = env.encode(3,1,2,2)

env.s = state
env.render()

#%%
"""
Aksiyonlar:
        0: güney
        1: kuzey
        2: doğu
        3: batı
        4: yolcu al
        5: yolcu indir
"""
#olasılık, sonraki state, 
print(env.P[331])


#%%

env.reset()
step = 0
total_reward = 0
list_visualize = []
while True:
    step +=1
    #action seç
    action = env.action_space.sample()
    
    #actionu gerçekleştir ve rewardı al
    state, reward, done, _ =  env.step(action) #state = next_state
   
    #Ödülü Al
    total_reward += reward
    
    #Görselleştir
    list_visualize.append({"frame": env,
                           "state": state,
                           "action": action,
                           "reward": reward,
                           "Total_Reward": total_reward})
    
    env.render()
    if done:
        break
    
#%%
from time import sleep        
for i, frame in enumerate(list_visualize):
    print(frame["frame"])
    print("Timestep: ",i+1)
    print("State: ",frame["state"])
    print("Action: ",frame["action"])
    print("Reward: ",frame["reward"])
    print("Total Reward: ",frame["Total_Reward"])
    sleep(1)
     