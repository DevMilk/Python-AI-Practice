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
        pass

    def build_model(self):
        #Build NN
        pass
    def store(self, state, action, reward, next_state, done):
        #Storage
        pass
    def act(self,state):
        #execute action
        pass
    def replay(self,batch_size):
        #training
        pass
    def adaptiveEGreedy(self):
        pass
        

if __name__ == "__main__":
    
    
    #init env and agent
    iterations = 100
    
    
    for i in range(iterations):
        
        #init environment
        
        Done = False
        
        while(not Done):
            
            #act
            
            #step
            
            #remember
            
            #update state
            
            #replay
            
            #adjust epsilon


