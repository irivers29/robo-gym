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
import torch

from stable_baselines3 import TD3
from stable_baselines3.td3 import MlpPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import matplotlib.pyplot as plt


target_machine_ip = '127.0.0.1' # or other machine 'xxx.xxx.xxx.xxx'

best_mean_reward, n_steps = -np.inf, 0
num_best_model = 0
time_steps = 4000000

def main(args):

    env = gym.make('ShelfEnvironmentPositioningURSim-v0', ur_model='ur10e', ip=target_machine_ip, gui=False)
    env = ExceptionHandling(env)
    log_dir='mon/'
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    obs = env.reset()

    observation_dim = len(obs)
    #achieved_goal_dim = len(info['ee_coord'])
    #desired_goal_dim = len(info['target_coord'])
    action_dim = env.action_space.shape[0]
    state_dim = observation_dim 
    #assert achieved_goal_dim == desired_goal_dim

    print("observation", obs)
    #print("ee_coord", info["ee_coord"])
    #print("target coordinates", info["target_coord"])
    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    print("-----------------------")
    print('Parameters:')
    print("Observation Size:", observation_dim)
    #print("Goal Size:", achieved_goal_dim)
    print("State Size:", observation_dim)
    print("Action Size:", action_dim)
    print("State Size:", state_dim)
    print("-----------------------")

    callback = SaveOnBestTrainingRewardCallback(check_freq=2000, log_dir=log_dir)
    
    #model = TD3(MlpPolicy, env, verbose=1, device='cuda', tensorboard_log="./TD3_positive_reward_tensorboard/")
    model = TD3.load("mon/21_02_pos_results/best_model")
    model.set_env(env)
    model.learn(total_timesteps=args['num_episodes'], callback=callback)
    model.save('TD3')
    plot_results([log_dir], args["num_episodes"], results_plotter.X_TIMESTEPS, "TD3")


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr_actor', help='actor learning rate', default=0.0001)
    parser.add_argument('--lr_critic', help='critic learning rate', default=0.001)
    parser.add_argument('--batch_size', help='batch size', default=64)
    parser.add_argument('--num_episodes', help='episodes to train', default=4000000)
    parser.add_argument('--episodes-length', help='max length for one episode', default=1000)
    parser.add_argument('--HER', help='Hinsight Experience Replay', default=False)

    args = vars(parser.parse_args())

    start_time = datetime.datetime.now()
    main(args)
    print("---%s seconds---"%(datetime.datetime.now() - start_time))