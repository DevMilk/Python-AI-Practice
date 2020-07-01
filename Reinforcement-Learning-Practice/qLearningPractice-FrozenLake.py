# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 22:46:21 2020

@author: Ugur
"""

import gym, random
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v0").env 

 


#Q table
q_table = np.zeros([env.observation_space.n,env.action_space.n])

#Hyperparameter
alpha = 0.1
gamma = 0.9
epsilon = 0.1

#Plot Matrix
reward_list   = []
dropouts_list = [] 

#%%
episodeNum = 5000 
i = 1
while(i<episodeNum ):
    
    Done = False
    # initialize 
    state = env.reset()
    
    reward_count = 0
    dropouts = 0
     
    while(not Done):
        
        #exploit or explore to find action
        if (random.uniform(0,1) < epsilon):
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        #action process and take reward
        
        next_state, reward, Done, _ = env.step(action)
        
        #Q learning Function 
        old = q_table[state,action]
        next_max = np.max(q_table[next_state])
        next_value = (1-alpha)*old + alpha*(reward + gamma*next_max)
        
        #Q table update
        q_table[state, action] = next_value
        
        #update state
        state = next_state 
        
        #find wrong dropouts
        if reward == -10:
            dropouts += 1
        reward_count += reward    
    i += 1
    
    if(i%10==0):
        dropouts_list.append(dropouts)
        reward_list.append(reward_count)
        print("Episode: {}, reward: {}, wrong dropout {}".format(i,reward_count,dropouts))     
#%% Viualize
        
fig, ax = plt.subplots(1,2) 
ax[0].plot(reward_list)
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Reward")

ax[1].plot(dropouts_list)
ax[1].set_xlabel("Iteration")       
ax[1].set_ylabel("Dropouts")

plt.show


#%% ınterpret Q Table 

"""
Aksiyonlar:
        0: güney
        1: kuzey
        2: doğu
        3: batı
        4: yolcu al
        5: yolcu indir
"""


"""Q tableda 197. satırdaki kararlar çok avantajlı
 Bu satırın hangi stateye karşılık geldiğini ve bu statedeki mantıklı actionun ne oldığğunu
 o stateye ulaşarak anlayabiliriz. """
def Decode(env,stateNum):
    arr = []
    for i in env.decode(stateNum):
        arr.append(i)
    arr.reverse()   
    return arr 
env.s = env.encode(*Decode(env,14))
env.render()