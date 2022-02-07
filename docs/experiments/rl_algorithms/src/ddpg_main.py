from os import stat
from ddpg import Agent
import gym
import numpy as np
import os, sys, time
from utils import utils


env = gym.make('LunarLanderContinuous-v2')
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001,
                env=env, batch_size=64, layer1_size=400, layer2_size=300,
                n_actions=2)


def unpack_observation(observation, info):
    
    achieved_goal = info['ee_cord']
    target_goal = info['target_coord']
    state = np.concatenate((observation, info['ee_coord']))
    state_prime = np.concatenate((observation, info['target_coord']))

    return achieved_goal, target_goal, state, state_prime



np.random.seed(0)
score_history = []
for i in range(1000):                   #NUMBER OF EPISODES

    done = False
    score = 0
    episode_maximum_score = 0
    obs, info = env.reset()

    achieved_goal, desired_goal, s, s_prime = unpack_observation(obs, info)

    while not done:                     #INCLUDE MAXIMUM LENGTH FOR EACH EPISODE (ALREADY INCLUDED IN REWARD COMPUTING)
        
        act = agent.choose_action(obs)  #DOES IT HAVE THE RIGHT SHAPE/LENGTH? + NOISE IS ALREADY INCLUDED
        
        obs_next, reward, done, info = env.step(act)
        achieved_goal, desired_goal, state_next, state_prime_next = unpack_observation(obs_next, info)
    
        agent.remember(s, act, reward, state_next, int(done))

        #HINDSIGHT EXPERIENCE REPLAY
        substitute_goal = achieved_goal.copy
        substitute_reward, done_substitute, info_substitute = env.compute_reward_HER(achieved_goal, substitute_goal, info['rs_state'])
        agent.remember(s_prime, act, substitute_reward, done_substitute, state_prime_next)

        if agent.get_buffer_size() > int(args['batch_size']):

            agent.learn()

        score += reward
        obs = state_next

    score_history.append(score)
    print('episode', i, 'score %.2f' %score,
     '100 game average %.2f' %np.mean(score_history[-100:]))
    if i % 25 == 0:
        agent.save_models()