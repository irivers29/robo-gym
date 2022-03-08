from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import time
import argparse
import os
import datetime

import gym
from matplotlib.pyplot import plot
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.td3 import MlpPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import matplotlib.pyplot as plt


target_machine_ip = '127.0.0.1' # or other machine 'xxx.xxx.xxx.xxx'

env = gym.make('ShelfEnvironmentPositioningURSim-v0', ur_model='ur10e', ip=target_machine_ip, gui=True)
env = ExceptionHandling(env)
log_dir='mon/'
env = Monitor(env, log_dir)

model = TD3.load("mon/21_02_pos_results/best_model")

for i in range(100):
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)