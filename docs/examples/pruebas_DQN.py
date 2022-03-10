import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
import numpy as np
from matplotlib import pyplot as plt
import cv2
from trac_ik_python.trac_ik import IK

target_machine_ip = '127.0.0.1' # or other machine 'xxx.xxx.xxx.xxx'

# initialize environment
env = gym.make('ShelfEnvironmentDQNSim-v0', ur_model='ur10e', ip=target_machine_ip, gui=True)
env = ExceptionHandling(env)


msg_height, msg_width = 480, 640
def axis_angle_to_quat(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    
    return qx,qy,qz,qw



for episode in range(10):
    done = False
    print("prepared for reset:", episode)
    env.reset()
    print("reseted")
    while not done:
        step_space = env.action_space.sample()
        save_image = True
        

        #step_space = env.action_space.sample()
        #print(env.action_space.sample())
        #step_space = np.array([0.7519534, 0.46655673, 0.70558476, -0.6050794, -0.7676735, 0.5441871,  0.1002852])
        state, reward, done, info = env.step(step_space)
        done = False


