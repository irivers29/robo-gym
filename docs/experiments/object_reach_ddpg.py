import argparse
import copy
import os
import sys
import time
from datetime import datetime
from turtle import shape

import cv2
import gym
import numpy as np
import robo_gym
from matplotlib import pyplot as plt
from robo_gym.wrappers.exception_handling import ExceptionHandling

from rl_algorithms.src.ddpg import Agent
from rl_algorithms.utils.utils import utils

target_machine_ip = '127.0.0.1' # or other machine 'xxx.xxx.xxx.xxx'

def main(args):
    env = gym.make('ShelfEnvironmentPositioningURSim-v0', ur_model='ur10e', ip=target_machine_ip, gui=True)
    env = ExceptionHandling(env)

    utils_ddpg = utils(name='FirstAttempt')

    obs, info = env.reset()
    

    observation_dim = len(obs)
    achieved_goal_dim = len(info['ee_coord'])
    desired_goal_dim = len(info['target_coord'])
    action_dim = env.action_space.shape[0]
    state_dim = observation_dim 
    assert achieved_goal_dim == desired_goal_dim

    print("observation", obs)
    print("ee_coord", info["ee_coord"])
    print("target coordinates", info["target_coord"])

    print("-----------------------")
    print('Parameters:')
    print("Observation Size:", observation_dim)
    print("Goal Size:", achieved_goal_dim)
    print("State Size:", observation_dim)
    print("Action Size:", action_dim)
    print("State Size:", state_dim)
    print("-----------------------")

    HER = args['HER']
    agent = Agent(alpha=float(args['lr_actor']), beta=float(args['lr_critic']),
                 input_dims= [state_dim], tau=0.001, env=env, 
                 batch_size=int(args['batch_size']), layer1_size=400, 
                 layer2_size=300, n_actions=action_dim)

    np.random.seed(0)
    score_history = []
    episode_scores = []

    for episode in range(args['num_episodes']):

        """
        state := current state (observation + achieved goal)
        state_next := state achieved after a step
        state_prime_next := state that should be achieved after a state
        obs := observation of environment
        obs_next := observation of environment after step
        """
        
        done = False
        score = 0
        episode_maximum_score = 0

        obs, info = env.reset()


        #achieved_goal, desired_goal, state, s_prime = unpack_observation(obs, info)
        i = 0

        state = obs
        
        while not done:

            act = agent.choose_action(state)
            
            print(state)
        
            obs_next, reward, done, info = env.step(act)

            #print("obs next", obs_next)

            #achieved_goal, desired_goal, state_next, state_prime_next = unpack_observation(obs_next, info)

            #print("achieved goal", achieved_goal)
            #print("desired goal", desired_goal)
            #print("state next", state_next)
            #print("state prime next", state_prime_next)

            agent.remember(state, act, reward, obs_next, int(done))

            #HINDSIGHT EXPERIENCE REPLAY
            if HER:
                substitute_state = change_state(state, info)
                
                achieved_goal = info["ee_coord"]
                substitute_goal = copy.deepcopy(achieved_goal)
                substitute_reward, done_substitute, info_substitute = env.compute_reward_HER(achieved_goal, substitute_goal, info['rs_state'])

                #print("s_prime", s_prime)
                agent.remember(substitute_state, act, substitute_reward, obs_next, int(done_substitute))
        
            if agent.get_buffer_size() > int(args['batch_size']):

                agent.learn()
            #print(info)
            score += reward
            state = obs_next
            score_history.append(score)
            i += 1
        
        episode_scores.append(score)
        #print('episode', episode, 'score %.2f' %score, '100 game average %.2f' %np.mean(score_history[-100:]), 'final status:', info['final_status'])
        print('episode', episode, 'score %.2f' %score, '100 game average %.2f' %i, 'final status:', info['final_status'])
        if episode % 2 == 0:
            utils_ddpg.reward_graphic(episode_scores)
            agent.save_models()

def unpack_observation(observation, info):
    """
    achieved_goal := position of endeffector
    target_goal := position of object
    state := observation + achieved_goal
    state_prime := observation + target_goal
    
    """
    achieved_goal = info['ee_coord']
    target_goal = info['target_coord']
    state = np.concatenate((observation, info['ee_coord']))
    state_prime = np.concatenate((observation, info['target_coord']))

    return achieved_goal, target_goal, state, state_prime

def change_state(observation, info):

    state = copy.deepcopy(observation)
    state[0:3] = info["ee_coord"]
    return state

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr_actor', help='actor learning rate', default=0.0001)
    parser.add_argument('--lr_critic', help='critic learning rate', default=0.001)
    parser.add_argument('--batch_size', help='batch size', default=64)
    parser.add_argument('--num_episodes', help='episodes to train', default=2000)
    parser.add_argument('--episodes-length', help='max length for one episode', default=1000)
    parser.add_argument('--HER', help='Hinsight Experience Replay', default=False)

    args = vars(parser.parse_args())

    start_time = time.time()
    main(args)
    print("---%s seconds---"%(time.time() - start_time))
