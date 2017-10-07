#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:30:18 2017

@author: junliu
"""

import gym
import numpy as np
#%%
env = gym.make('CartPole-v0') 

def evaluate_given_parameter_sigmoid(env, weight):
    #启动初始状态
    observation = env.reset()
    #这组参数返回的总reward
    total_reward = 0.
    for t in range(1000):
        #这个渲染函数就是实时展示图1，如果隐藏，代码正常执行，但是不会显示图1了
        #env.render()
        _, action = choose_action(weight, observation)
    
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

def choose_action(weight, observation):
    weighted_sum = np.dot(weight, observation)
    pi = 1 / (1 + np.exp(-weighted_sum))
    if pi > 0.5:
        action = 1
    else:
        action = 0
    return pi, action
    
def generate_episode(env, weight):
    episode = []
    pre_observation = env.reset()
    t = 0
    #generate 1 episodes for training.
    while 1:
        #env.render()
        pi, action = choose_action(weight, pre_observation)
    
        observation, reward, done, info = env.step(action)
        #将这个episode的每一步产生的数据保存下来
        episode.append([pre_observation, action, pi, reward])
        pre_observation = observation
    
        t += 1
        if done or t > 1000:
            break
    return episode

def monte_carlo_policy_gradient(env):
    learning_rate = -0.0001
    best_reward = -100.0
    weight = np.random.rand(4)
    for iiter in xrange(10000):
        cur_episode = generate_episode(env, weight)
        for t in range(len(cur_episode)):
             
            observation, action, pi, reward = cur_episode[t]
            #根据第七课的更新公式
            weight += learning_rate*(1-pi)*np.transpose(-observation)*reward
    #衡量算出来的weight表现如何
    cur_reward = evaluate_given_parameter_sigmoid(env, weight)
    print 'Monte-Carlo policy gradient get reward', cur_reward

#%%
monte_carlo_policy_gradient(env)