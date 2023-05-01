

from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import gym
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def opening_sec ():
    mu, sigma = 5, 1 # mean and standard deviation
    s = np.random.normal(mu, sigma, 1)
    return s[0]
def opening_event_func(weather, hour):
    weather_lambda = {'spring':3, 'summer':4, 'autumn':2, 'winter':1}
    hourly_lambda = [10 if (i < 20 and i > 10) else 5 for i in range(0, 25)]
    open_event = np.random.poisson(weather_lambda[weather]*hourly_lambda[hour], size=5)
    a = np.random.randint(0, 60*60-10, open_event[0])
    a.sort()
    opening_list = []
    hour_event = [0 for i in range(60*60)]
    for i in a:
        n = round(opening_sec())
        for j in range(i, i+n):
            hour_event[j] = 1
    return hour_event
    

class RefriEnv(Env):
    def __init__(self, weather):
        #action space
        custom_action_space = Discrete(4)
        self.action_space = custom_action_space
        # The real values are that mapping the action value
        # 0 to 3 is mapping the -5,0,5,10
        self.action_to_value = {0: -5, 1: 0, 2: 5, 3: 10}
        # Temperature array
        self.observation_space = Box(low=np.array([-50]), high=np.array([50]))
        # Set start temp
        self.state = 0 + random.randint(-3,3)
        # Set refrigerator length
        self.refri_hour = 0
        self.refri_min = 0
        self.refri_sec = 0
        self.weather = weather

        
        self.hour_event = opening_event_func(self.weather, self.refri_hour)

        #DR 발생에 대한 boolean 값
        self.event_DR = False
        #DR 발생 시간에 대한 random int 값
        self.event_DR_time = random.randint(13,18)

        #임시로 설정
        self.U = 200
        self.A = 1.5
        self.T_ext = 25
        self.h = 10
        self.m = 10
        self.c = 3900
        self.P_cool = 200
        self.E = 0.5


    #transform the action to value
    def map_action_to_value(self, action):
        return self.action_to_value[action]
        
    #transform the value to action
    def map_value_to_action(self, value):
        for action, val in self.action_to_value.items():
            if val == value:
                return action
        raise ValueError("Invalid value: {}".format(value))
    
    #RefriEnv 클래스의 step 메소드를 수정했음. 
    #이 메소드에서 시간대별 문 여닫는 이벤트 수를 고려하여 냉장고의 온도가 업데이트되는 방식을 변경함. 
    #먼저, 현재 시간대에서 발생하는 이벤트 수를 계산하고, 이 값을 O(t)로 사용
    def step(self, action):
        # Get current minute
        # Get door opening count for the current minute
        O = self.hour_event[self.refri_sec + self.refri_min * 60]

        # Apply action
        if np.any(self.state > action):
            self.state += (self.P_cool*self.E-self.U*self.A*(self.state-self.T_ext)+O*self.h*(self.T_ext-self.state))/(self.m*self.c)
        else:
            self.state += (-self.U*self.A*(self.state-self.T_ext)+O*self.h*(self.T_ext-self.state))/(self.m*self.c)
        
        # Increase refrigerator length by 1 second
        self.refri_sec += 1
        if self.refri_sec == 60:
            self.refri_sec = 0
            self.refri_min += 1
            
        if self.refri_min == 60:
            self.refri_min = 0
            self.refri_hour += 1
            self.hour_event = opening_event_func(self.weather, self.refri_hour)
        
        # Calculate reward
        if np.any(self.state > action): 
            reward = -0.01 
        else: 
            reward= -(self.state-action)*0.1
        
        # Check if episode is done
        if self.refri_hour == 24: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    def reset(self):
        # Reset refrigerator temperature
        self.state = 0 + random.randint(-3,3)
        # Reset refrigerator time
        self.refri_hour = 0
        self.refri_min = 0
        self.refri_sec = 0
        # Reset door opening event count
        self.hour_event = opening_event_func(self.weather, self.refri_hour)
        return self.state
    
env = RefriEnv("spring")

env.observation_space.sample()

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        #env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))



episodes = 1
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    x_val1 = []
    x_val2=[]
    y_val1 = []
    y_val2 = []
    a=0
    action1=[]
    n_state1=[]

    plt.style.use('fivethirtyeight')
    while not done:
        #env.render()
        action = env.action_space.sample()
        action1.append(action)
        n_state, reward, done, info = env.step(action)
        n_state1.append(n_state)
        a=a+1
        
    
    index=count()
    def animate1(i):
            x_val1.append(next(index))
            y_val1.append(action1[i])
           
            plt.figure(1)
            plt.cla()
            plt.plot(x_val1, y_val1,label='action',color='r')
            plt.xlabel('time (sec)')
            plt.ylabel('Temperature(°C)')
            plt.legend(loc = 'upper left')
            plt.tight_layout()
            
    

    def animate2(i):
            x_val2.append(next(index))
           
            y_val2.append(n_state1[i])
            plt.figure(2)
            plt.cla()
           
            plt.plot(x_val2, y_val2,label='state',color='g')
            plt.xlabel('time (sec)')
            plt.ylabel('Temperature(°C)')
            plt.legend(loc = 'upper left')
            plt.tight_layout()
    
    
    ani1 = FuncAnimation(plt.figure(1), animate1, interval = 100)
    ani2 = FuncAnimation(plt.figure(2), animate2, interval = 100)
    score+=reward
    plt.tight_layout()
    plt.show()  

    
    print('Episode:{} Score:{}'.format(episode, score))
 

 

 
 

