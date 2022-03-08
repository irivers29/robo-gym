import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
import numpy as np
from matplotlib import pyplot as plt
import cv2

target_machine_ip = '127.0.0.1' # or other machine 'xxx.xxx.xxx.xxx'

# initialize environment
env = gym.make('ShelfEnvironmentImMIPositioninsURSim-v0', ur_model='ur10e', ip=target_machine_ip, gui=True)
env = ExceptionHandling(env)

msg_height, msg_width = 480, 640

def _image_to_numpy(byte_image, save_image, episode) -> np.array:
    "reconvert image in bytes to numpy array"
    np_image = np.frombuffer(byte_image, dtype=np.uint8)
    np_image = np_image.reshape((480,640,3))
    np_image = np_image[0:388, 227:590]
    #print(np_image.dtype)
    plt.imshow(np_image)
    plt.show()
    

    return np_image
def _depth_to_numpy(byte_depth) -> np.array:
    im = np.ndarray(shape=(msg_height, msg_width),
                           dtype=np.uint16, buffer=byte_depth)
    im = im.reshape((480,640,1))
    image = im.copy()
    image = image*255.0/image.max()

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


