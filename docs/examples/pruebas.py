import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
import numpy as np
from matplotlib import pyplot as plt
import cv2

target_machine_ip = '127.0.0.1' # or other machine 'xxx.xxx.xxx.xxx'

# initialize environment
env = gym.make('ShelfEnvironmentURSim-v0', ur_model='ur10e', ip=target_machine_ip, gui=True)
env = ExceptionHandling(env)

def _image_to_numpy(byte_image) -> np.array:
    "reconvert image in bytes to numpy array"
    np_image = np.frombuffer(byte_image, dtype=np.uint8)
    np_image = np_image.reshape((480*2,640*2,3))
    #plt.imshow(np_image)
    #plt.show()

    return np_image

num_episodes = 10

for episode in range(num_episodes):
    done = False
    print("prepared for reset:", episode)
    env.reset()
    print("reseted")
    i = 1
    #while not done:
    step_space = env.action_space.sample()
    print(step_space)
    while i == 1:
        pass
        # random step in the environment
        #if i == 1:
        #    step_space = env.action_space.sample()
        np_image = _image_to_numpy(env.get_image())
        #step_space = env.action_space.sample()
        #print(env.action_space.sample())
        #step_space = np.array([0.7519534, 0.46655673, 0.70558476, -0.6050794, -0.7676735, 0.5441871,  0.1002852])
        state, reward, done, info = env.step(step_space)


