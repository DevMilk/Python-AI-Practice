# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:55:29 2020

@author: Ugur
"""

import gym, random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam 


# Actionss
#Type: Discrete (2)
#0 : Push cart to left
#1 : Push cart to right

#Observations
#Type: Box (4)
#0 : Cart Position: -2.4 to 2.4
#1 : Cart Velocity: -inf to inf
#2 : Pole Angle   : -41.8 to 41.8 degree
#3 : Pole Velocity At Tip: -inf to inf  

#Reward
# 1 for every step taken, including termination 


class Agent:
    def __init__(self,env):
        #parameters- hyperparameters
        self.state_size   = env.observation_space.shape[0] 
        self.action_space = env.action_space 
        self.gamma = 0.95
        self.lr = 0.001
        
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen=10000)
        self.model = self.build_model() 

    def build_model(self):
        #Build NN
        model = Sequential()
        model.add(Dense(32, input_dim= self.state_size, activation = "tanh")) 
        model.add(Dense(4,activation="linear"))
        model.add(Dense(self.action_space.n,activation="linear"))
        model.compile(Adam(learning_rate = self.lr ),loss="mse",metrics=["accuracy"])
        print(model.summary())
        return model 
    def store(self, state, action, reward, next_state, done):
        #Storage
        self.memory.append((state, action, reward, next_state, done))
         
    def act(self,state):
        #execute action
        if(random.uniform(0,1) <= self.epsilon):
            return self.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
        
    def replay(self,batch_size):
        #training
        if(len(self.memory)< batch_size):
            return
        
        minibatch = random.sample(self.memory,batch_size)
        
        for state,action, reward,next_state, done in minibatch:  
            
            if done:
                target = reward
            else: 
                target = reward + self.gamma*np.amax(self.model.predict(next_state))  
            
            train_target = self.model.predict(state)
            train_target[0][action] = target 
            
            self.model.fit(state,train_target,verbose= 0)
        
    def adaptiveEGreedy(self):
        if(self.epsilon > self.epsilon_min):
            self.epsilon *= self.epsilon_decay   
        
#%%
if __name__ == "__main__":
    
    
    #init env and agent
    env = gym.make("LunarLander-v2").env 
    agent = Agent(env)
    iterations = 30
    batch_size = 16
    state_size = env.observation_space.shape[0]
    for i in range(iterations):
        
        #init environment
        state = env.reset() 
        state = np.reshape(state,[1,state_size])
        
        counter = 0
        Done = False
        
        while(not Done):
            
            
            #act
            action = agent.act(state)
            
            #step
            next_state, reward, Done, _ = env.step(action)
            next_state = np.reshape(next_state,[1,state_size])
            #store 
            agent.store(state,action,reward,next_state,Done)
           
            #update state
            state = next_state
            
            #replay
            agent.replay(batch_size)
          
            #adjust epsilon
           
            agent.adaptiveEGreedy()
            counter += 1
            
            print("Iteration: ",i," time: ",counter)


#%% Visualize
import time
trained_model = agent
state = env.reset()
state = np.reshape(state,[1,state_size])
time_t = 0
Done =False
while(not Done):
    env.render()
    action = trained_model.act(state)
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state,[1,state_size])
    state = next_state
    time_t += 1
    print(time_t)
    time.sleep(0.01)

print("Done")    
    