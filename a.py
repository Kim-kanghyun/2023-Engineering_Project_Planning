from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import gym
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


class RefriEnv(Env):
    def __init__(self, weather):
        # action space
        custom_action_space = Box(low=-20.0, high=8.0, shape=(1,), dtype=float)
        self.action_space = custom_action_space
        # Temperature array
        self.observation_space = Box(low=np.array([-50]), high=np.array([50]))
        # Set start temp
        self.state = 0 + random.randint(-3, 3)
        # Set refrigerator length
        self.refri_hour = 0
        self.refri_min = 0
        self.refri_sec = 0

        self.weather = weather
        self.hour_event = opening_event_func(self.weather, self.refri_hour)
        # DR 발생에 대한 boolean 값
        self.event_DR = False
        # DR 발생 시간에 대한 random int 값
        self.event_DR_time = random.randint(13, 18)

        # 임시로 설정
        self.U = 200
        self.A = 1.5
        self.T_ext = 25
        self.h = 10
        self.m = 10
        self.c = 3900
        self.P_cool = 200
        self.E = 0.5

    # RefriEnv 클래스의 step 메소드를 수정했음. 
    # 이 메소드에서 시간대별 문 여닫는 이벤트 수를 고려하여 냉장고의 온도가 업데이트되는 방식을 변경함. 
    # 먼저, 현재 시간대에서 발생하는 이벤트 수를 계산하고, 이 값을 O(t)로 사용
    def step(self, action):
        # Get current minute
        # Get door opening count for the current minute
        O = self.hour_event[self.refri_sec + self.refri_min * 60]

        # #action samping(-20.5 ~ 5.0 one decimals float number)
        # action = self.action_space.sample()
        action = np.round(action, decimals=1)

        # 오후 1시에서 6시 사이(최대부하로 냉장고를 가동하는 시간)에 랜덤으로 주어지는 DR 발생시간을 계산하여 DR상태를 일으킨다.
        # DR 상황은 1시간동안 지속된다.
        if self.event_DR_time == self.refri_hour:
            self.event_DR = True
        elif self.event_DR_time + 1 == self.refri_hour:
            self.event_DR = False
        else:
            self.event_DR = False

        cool_state = (self.P_cool * self.E + self.U * self.A * (self.state - self.T_ext) + O * self.h * (
                self.T_ext - self.state)) / (self.m * self.c)
        normal_state = (-self.U * self.A * (self.state - self.T_ext) + O * self.h * (self.T_ext - self.state)) / (
                self.m * self.c)

        def condition (a, b):
            if action < a or action > b:
                reward = -0.1
            if np.any(self.state > action):
                diff_state = cool_state
            else:
                diff_state = normal_state
            self.state += diff_state
            return diff_state


        # Apply action
        # DR이 발생되었을 때의 처리
        # DR 상황일 때 Th보다 action(설정온도)이/가 작으면 reward의 손실을 준다.
        if self.event_DR:
            power_usage = condition(0, 5)
            power_usage_fee = power_usage * 5
        # DR이 발생되지 않았을 때의 상황
        # 전력 사용시간대에 따라 Ts를 달리 생각하여 reward값을 부여한다.
        else:
            # 경부하 시간대
            if 22 <= self.refri_hour or self.refri_hour <= 8:
                power_usage = condition(-5, 0)
                power_usage_fee = power_usage * 0.5
            # 중간부하 시간대
            elif ((8 <= self.refri_hour <= 11) or
                  (12 <= self.refri_hour <= 13) or
                  (18 <= self.refri_hour <= 22)):
                power_usage = condition(-3, 2)
                power_usage_fee = power_usage * 2
            # 최대부하 시간대
            else:
                power_usage = condition(-1, 4)
                power_usage_fee = power_usage * 3

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
            reward = -(self.state - action) * 0.1

        # Check if episode is done
        if self.refri_hour == 24:
            done = True
        else:
            done = False

        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, power_usage, power_usage_fee, done, info

    def render(self):
        # Implement viz
        pass

    def reset(self):
        # Reset refrigerator temperature
        self.state = 0 + random.randint(-3, 3)
        # Reset refrigerator time
        self.refri_hour = 0
        self.refri_min = 0
        self.refri_sec = 0
        # Reset door opening event count
        self.hour_event = opening_event_func(self.weather, self.refri_hour)
        return self.state


env = RefriEnv("spring")

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
    x_val5 = []
    x_val6 = []
    y_val1 = []
    y_val2 = []
    y_val3 = []
    y_val4 = []
    y_val5 = []
    y_val6 = []

    a = 0
    action1 = []
    n_state1 = []
    reward1 = []
    score1 = []

    total_power_usage = []
    total_power_usage_fee = []

    sum_power_usage = 0
    sum_power_usage_fee = 0


    plt.style.use('fivethirtyeight')
    while not done:
        # env.render()
        if a % 60 == 0:
            action = env.action_space.sample()
        action1.append(action)
        n_state, reward, power_usage, power_usage_fee, done, info = env.step(action)
        n_state1.append(n_state)
        reward1.append(reward)
        sum_power_usage += max(-power_usage, 0)
        sum_power_usage_fee += max(-power_usage_fee, 0)
        total_power_usage.append(sum_power_usage)
        total_power_usage_fee.append(sum_power_usage_fee)
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

        plt.plot(x_val3, y_val3, label='reward', color='b')
        plt.xlabel('time (sec)')
        plt.ylabel('reward')
        plt.legend(loc='upper left')
        plt.tight_layout()


    def animate4(i):
        x_val4.append(next(index))

        y_val4.append(score1[i])
        plt.figure(4)
        plt.cla()

        plt.plot(x_val4, y_val4, label='score', color='purple')
        plt.xlabel('time (sec)')
        plt.ylabel('score')
        plt.legend(loc='upper left')
        plt.tight_layout()


    def animate5(i):
        x_val5.append(next(index))

        y_val5.append(total_power_usage[i])
        plt.figure(5)
        plt.cla()

        plt.plot(x_val5, y_val5, label='Power', color='black')
        plt.xlabel('time (sec)')
        plt.ylabel('watt')
        plt.legend(loc='upper left')
        plt.tight_layout()

    def animate6(i):
        x_val6.append(next(index))

        y_val6.append(total_power_usage_fee[i])
        plt.figure(6)
        plt.cla()

        plt.plot(x_val6, y_val6, label='fee', color='yellow')
        plt.xlabel('time (sec)')
        plt.ylabel('won')
        plt.legend(loc='upper left')
        plt.tight_layout()


    ani1 = FuncAnimation(plt.figure(1), animate1, interval=10)
    ani2 = FuncAnimation(plt.figure(2), animate2, interval=10)
    ani3 = FuncAnimation(plt.figure(3), animate3, interval=10)
    ani4 = FuncAnimation(plt.figure(4), animate4, interval=10)
    ani5 = FuncAnimation(plt.figure(5), animate5, interval=10)
    ani6 = FuncAnimation(plt.figure(6), animate6, interval=10)
    plt.tight_layout()
    plt.show()

    print('Episode:{} Score:{}'.format(episode, score))
