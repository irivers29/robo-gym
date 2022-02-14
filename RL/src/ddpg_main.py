from ddpg import Agent
import gym
import numpy as np


env = gym.make('LunarLanderContinuous-v2')


agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001,
                env=env, batch_size=64, layer1_size=400, layer2_size=300,
                n_actions=2)

np.random.seed(0)
score_history = []
for i in range(1000):                   #NUMBER OF EPISODES
    done = False
    score = 0
    obs = env.reset()

    ### UNPACK OBSERVATION INTO ACHIEVED_GOAL, DESIRED_GOAL, STATE, ...

    while not done:                     #INCLUDE MAXIMUM LENGTH FOR EACH EPISODE
        
        act = agent.choose_action(obs)  #DOES IT HAVE THE RIGHT SHAPE/LENGTH? + NOISE IS ALREADY INCLUDED
        
        new_state, reward, done, info = env.step(act)
        #UNPACK NEW_STATE INTO ACHIEVED_GOAL, DESIRED_GOAL, ...
        
        ##
        agent.remember(obs, act, reward, new_state, int(done))

        ##############################################################
        #ADD HINDSIGHT EXPERIENCE REPLAY
        substitute_goal = achieved_goal.copy
        substitute_reward = env.compute_reward(achieved_goal, substitute_goal, info)
        agent.remember()
        ##############################################################
        if replay_memory.size() > int(args['batch_size']):
            pass
        agent.learn()
        score += reward
        obs = new_state

    score_history.append(score)
    print('episode', i, 'score %.2f' %score,
     '100 game average %.2f' %np.mean(score_history[-100:]))
    if i % 25 == 0:
        agent.save_models()