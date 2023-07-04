from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import gym
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math


def opening_sec():
    mu, sigma = 5, 1  # mean and standard deviation
    s = np.random.normal(mu, sigma, 1)
    return s[0]


def opening_event_func(weather, hour):
    weather_lambda = {'spring': 3, 'summer': 4, 'autumn': 2, 'winter': 1}
    hourly_lambda = [10 if (20 > i > 10) else 5 for i in range(0, 25)]
    open_event = np.random.poisson(weather_lambda[weather] * hourly_lambda[hour], size=5)
    a = np.random.randint(0, 60 * 60 - 10, open_event[0])
    a.sort()
    opening_list = []
    hour_event = [0 for i in range(60 * 60)]
    for i in a:
        n = round(opening_sec())
        for j in range(i, i + n):
            hour_event[j] = 1
    return hour_event

#각 DR event에 따른 발생 시각을 조절하는 함수들
def standard_DR():
    hour = 12
    while hour == 12:
        hour = random.randint(9,20)
    min = random.randint(0,59)
    duration = random.randint(1,4)
    print('DR_hour:{} DR_min:{}'.format(hour, min))
    return hour, min, duration

def small_mid_DR():
    hour = 12
    while hour == 12:
        hour = random.randint(9,20)
    min = random.randint(0,59)
    duration = 1
    print('DR_hour:{} DR_min:{}'.format(hour, min))
    return hour, min, duration

def public_DR():
    hour = 12
    while hour == 12:
        hour = random.randint(6,18)
    min = random.randint(0,59)
    duration = 1
    print('DR_hour:{} DR_min:{}'.format(hour, min))
    return hour, min, duration

class RefriEnv(Env):
    def __init__(self, weather):
        # action space
        #custom_action_space = Box(low=-20.0, high=8.0, shape=(1,), dtype=float)
        custom_action_space = Discrete(280)
        self.action_space = custom_action_space
        # Temperature array
        self.observation_space = Box(low=np.array([-30]), high=np.array([20]))
        # Set start temp
        self.state = 0 + random.randint(-3, 3)
        # Set refrigerator length
        self.refri_hour = 0
        self.refri_min = 0
        self.refri_sec = 0

        self.weather = weather
        self.hour_event = opening_event_func(self.weather, self.refri_hour)
        self.DR_event = False
        print("1 : 표준 DR\n2 : 중소형 DR\n3 : 국민 DR")
        self.DR_version = input("Enter Demand Response type into number: ")
        if self.DR_version == '1':
            self.event_DR_hour, self.event_DR_min, self.duration_DR = standard_DR()
        elif self.DR_version == '2':
            self.event_DR_hour, self.event_DR_min, self.duration_DR = small_mid_DR()
        else:
            self.event_DR_hour, self.event_DR_min, self.duration_DR = public_DR()

        self.weather_fee = {'spring':[81.4, 88.8, 100.1], 'summer':[81.4, 132.6, 155.1],
                            'autumn':[81.4, 88.8, 100.1], 'winnter':[90.1, 120.5, 135.3]}
        #임시 dr_fee
        self.dr_fee = {'spring':[72.4, 79.8, 91.1], 'summer':[72.4, 123.6, 146.1],
                            'autumn':[72.4, 79.8, 91.1], 'winnter':[81.1, 111.5, 126.3]}

        # 임시로 설정
        self.U = 200
        self.A = 1.5
        self.T_ext = 25
        self.h = 10
        self.m = 10
        self.c = 3900
        self.E = 0.5

    # RefriEnv 클래스의 step 메소드를 수정했음. 
    # 이 메소드에서 시간대별 문 여닫는 이벤트 수를 고려하여 냉장고의 온도가 업데이트되는 방식을 변경함. 
    # 먼저, 현재 시간대에서 발생하는 이벤트 수를 계산하고, 이 값을 O(t)로 사용
    def step(self, action):
        # Get current minute
        # Get door opening count for the current minute
        O = self.hour_event[self.refri_sec + self.refri_min * 60]
        self.action = (float(action)-200)/10.0

        # #action samping(-20.5 ~ 8.0 one decimals float number)
        #DR 지속시간의 설정
        if self.event_DR_hour == self.refri_hour and self.event_DR_min == self.refri_min:
            self.DR_event = True
        if self.event_DR_hour + self.duration_DR == self.refri_hour and self.event_DR_min == self.refri_min:
            self.action = self.action
            self.DR_event = False
        else:
            self.action = self.action

        if self.state > self.action:
            self.P_cool = math.log2(self.state - self.action)
            self.P_ext = math.log2(self.T_ext - self.state)
            #U가 전관류율
            cool_state = (-self.P_cool * self.E + self.U * self.A * (self.state - self.T_ext) + O * self.h * (
                self.T_ext - self.state)) / (self.m * self.c)
        else:
            normal_state = (-self.U * self.A * (self.state - self.T_ext) + O * self.h * (self.T_ext - self.state)) / (
                self.m * self.c)
       
        #문열림 이벤트가 발생 했을 때, 질량을 줄인다. (m이 0이되면 식이 0으로 나눠지므로)
        if O==1 and self.m!=0.001:
            self.m+=-0.001
            
        def condition ():
            if np.any(self.state > self.action):
                diff_state = cool_state
                power = 100 + (self.P_cool / self.E) + (self.A * self.U / self.P_ext)
            else:
                diff_state = normal_state
                power = 100
            self.state += diff_state
            return power / 3600

        # Apply action
        # 경부하 시간대
        if 22 <= self.refri_hour or self.refri_hour <= 8:
            power_usage = condition()
            if self.DR_event == True:
                power_usage_fee = power_usage * self.dr_fee[self.weather][0]
            else:
                power_usage_fee = power_usage * self.weather_fee[self.weather][0]
        # 중간부하 시간대
        elif ((8 <= self.refri_hour <= 11) or
                (12 <= self.refri_hour <= 13) or
                (18 <= self.refri_hour <= 22)):
            power_usage = condition()
            if self.DR_event == True:
                power_usage_fee = power_usage * self.dr_fee[self.weather][1]
            else:
                power_usage_fee = power_usage * self.weather_fee[self.weather][1]
        # 최대부하 시간대(11~12시, 13~18시)
        else:
            power_usage = condition()
            if self.DR_event == True:
                power_usage_fee = power_usage * self.dr_fee[self.weather][2]
            else:
                power_usage_fee = power_usage * self.weather_fee[self.weather][2]

        # Increase refrigerator length by 1 second
        self.refri_sec += 1
        if self.refri_sec == 60:
            self.refri_sec = 0
            self.refri_min += 1

        if self.refri_min == 60:
            self.refri_min = 0
            self.refri_hour += 1
            self.hour_event = opening_event_func(self.weather, self.refri_hour)

        # Check if episode is done
        if self.refri_hour == 24:
            done = True
        else:
            done = False

        # Set placeholder for info
        info = {}
        #reward func
        # reward = (1-power_usage_fee)
        # if self.DR_event == True:
        #     reward = reward*10
        #     if self.state > 10 or self.state < 4:
        #         reward -= 10
        #     else:
        #         reward += 1
        # else:
        #     if self.state > 8 or self.state < 2:
        #         reward -= 1
        #     else:
        #         reward += 1

        #reward func 1
        # if self.state > 8 or self.state < 2:
        #     reward = -2
        # else:
        #     reward = (1-power_usage_fee)

        #reward func 2
        # weight = 0.3
        # if self.state > 8 or self.state < 2:
        #     reward = -2
        # else:
        #     reward = weight*(1-power_usage_fee) + (1-weight)*(self.state)

        #reward func 3 - 보류(안 될 확률이 매우 큼)
        # weight = 0.3
        # if self.state > 8 or self.state < 2:
        #     reward = -2
        # else:
        #     reward = weight * (1-power_usage_fee) + (1-weight) * (1-power_usage)

        # Return step information
        return self.state, reward, done, info

    def render(self ,mode='human',close=True):
        if self.refri_min%5==0 and self.refri_sec%60==0:
            print("action : ",self.action)
            print("state : ", self.state)   

    def reset(self):
        # Reset refrigerator temperature
        self.state = 0 + random.randint(-3, 3)
        # Reset refrigerator time
        self.refri_hour = 0
        self.refri_min = 0
        self.refri_sec = 0
        
        self.m = 10
        # Reset door opening event count
        #self.hour_event = opening_event_func(self.weather, self.refri_hour)
        return self.state


env = RefriEnv("summer")

env.observation_space.sample()

# episodes = 10
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0 

#     while not done:
#         #env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score+=reward
#     print('Episode:{} Score:{}'.format(episode, score))


episodes = 1
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    x_val1 = []
    x_val2 = []
    x_val3 = []
    x_val4 = []
    y_val1 = []
    y_val2 = []
    y_val3 = []
    y_val4 = []

    a = 0
    action1 = []
    n_state1 = []
    reward1 = []
    score1 = []

    plt.style.use('fivethirtyeight')
    while not done:
        # env.render()
        if a % 60 == 0:
            action = env.action_space.sample()
        action1.append(action)
        n_state, reward, done, info = env.step(action)
        n_state1.append(n_state)
        reward1.append(reward)
        score += reward
        score1.append(score)
        a = a + 1

    index = count()

    def animate1(i):
        x_val1.append(next(index))
        y_val1.append(action1[i])

        plt.figure(1)
        plt.cla()
        plt.plot(x_val1, y_val1, label='action', color='r')
        plt.xlabel('time (sec)')
        plt.ylabel('Temperature(°C)')
        plt.legend(loc='upper left')
        plt.tight_layout()


    def animate2(i):
        x_val2.append(next(index))

        y_val2.append(n_state1[i])
        plt.figure(2)
        plt.cla()

        plt.plot(x_val2, y_val2, label='state', color='g')
        plt.xlabel('time (sec)')
        plt.ylabel('Temperature(°C)')
        plt.legend(loc='upper left')
        plt.tight_layout()


    def animate3(i):
        x_val3.append(next(index))
    
        y_val3.append(reward1[i])
        plt.figure(3)
        plt.cla()
    
        plt.plot(x_val3, y_val3, label='reward(power fee)', color='b')
        plt.xlabel('time (sec)')
        plt.ylabel('reward (won)')
        plt.legend(loc='upper left')
        plt.tight_layout()
    
    
    def animate4(i):
        x_val4.append(next(index))
    
        y_val4.append(score1[i])
        plt.figure(4)
        plt.cla()
    
        plt.plot(x_val4, y_val4, label='score(summed power fee)', color='purple')
        plt.xlabel('time (sec)')
        plt.ylabel('score (won)')
        plt.legend(loc='upper left')
        plt.tight_layout()

    ani1 = FuncAnimation(plt.figure(1), animate1, interval=1)
    ani2 = FuncAnimation(plt.figure(2), animate2, interval=1)
    ani3 = FuncAnimation(plt.figure(3), animate3, interval=1)
    ani4 = FuncAnimation(plt.figure(4), animate4, interval=1)
    plt.tight_layout()
    plt.show()

    print('Episode:{} Score:{}'.format(episode, score))