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

msg_height, msg_width = 480, 640

def _image_to_numpy(byte_image, save_image, episode) -> np.array:
    "reconvert image in bytes to numpy array"
    np_image = np.frombuffer(byte_image, dtype=np.uint8)
    np_image = np_image.reshape((720,1280,3))
    
    if save_image:
        plt.imshow(np_image)
        plt.savefig('./images/shelf' + str(episode) + '.png')
    #plt.imshow(np_image)
    #plt.show()

    return np_image
def _depth_to_numpy(byte_depth) -> np.array:
    im = np.ndarray(shape=(msg_height, msg_width),
                           dtype=np.uint16, buffer=byte_depth)
    
    #plt.imshow(im)
    #plt.colorbar()
    #plt.show()
    #print(byte_depth)

num_episodes = 20

for episode in range(num_episodes):
    done = False
    print("prepared for reset:", episode)
    env.reset()
    print("reseted")
    while not done:
        step_space = env.action_space.sample()
        save_image = True
        
        np_image = _image_to_numpy(env.get_image(), save_image, episode)
        np_depth = _depth_to_numpy(env.get_depth())

        #step_space = env.action_space.sample()
        #print(env.action_space.sample())
        #step_space = np.array([0.7519534, 0.46655673, 0.70558476, -0.6050794, -0.7676735, 0.5441871,  0.1002852])
        state, reward, done, info = env.step(step_space)
        done = True


