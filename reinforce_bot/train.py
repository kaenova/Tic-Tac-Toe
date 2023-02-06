#! python

# Disable tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()
tf.get_logger().setLevel("CRITICAL")
tf.autograph.set_verbosity(0)

from agent import DDQNAgent
from game_env import Game
from copy import deepcopy
import time
from tqdm import tqdm

# For saving any python vars
import bz2
import pickle
import _pickle as cPickle
# ref: https://betterprogramming.pub/load-fast-load-big-with-compressed-pickles-5f311584507e
def compressed_pickle(title, data):
    with bz2.BZ2File("./" +title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)


if __name__=='__main__':
    # Initialize Game, number of game, and the agent to against eachother
    env = Game()
    n_game = 10000
    agent1 = DDQNAgent(alpha=0.003, gamma=0.999, 
                       epsilon=1, epsilon_end=0.1, epsilon_dec=0.99999,
                       input_dims=env.num_state, n_actions=env.num_action,
                       batch_size=64) # Agen mengisi O
    # agent2 = DDQNAgent(alpha=0.003, gamma=0.999, epsilon=1,  epsilon_end=0.1, epsilon_dec=0.999,
    #                 input_dims=env.num_state, n_actions=env.num_action,
    #                 batch_size=64) # Agen mengisi X
    
    # Initialize agent performance logging
    scores = {
        1: 0,
        2: 0
    }
    current_epsilon = {
        1: 1,
        2: 1
    }
    
    # Initialize time logging
    time_hist = []
    time_started = time.time()
    
    # Training part
    pbar = tqdm(total=n_game)
    for i in range(1, n_game):
        start = time.time()
        ### Initialize Scoring
        done = False
        score1 = 5
        score2 = 5
        observation = env.reset()
        while not done:
            action = agent1.choose_action(observation)
            observation_, reward, done = env.step(action)
            score1 += reward
            if done: score1 = reward
            agent1.remember(observation, action, score1, observation_, done)
            """
            if env.round % 2 == 0:
                action = agent1.choose_action(observation)
                observation_, reward, done = env.step(action)
                score1 += reward
                if done: score1 = reward
                agent1.remember(observation, action, score1, observation_, done)
            else:
                action = agent2.choose_action(observation)
                observation_, reward, done = env.step(action)
                score2 += reward
                if done: score2 = reward
                agent2.remember(observation, action, score2, observation_, done)
            """
            
            if env.round == 9: # Ketika seri
                score1 = -1
                score2 = 5
                
            
            observation = observation_
            
        agent1.learn()
        # agent2.learn()
        
        # Logging
        current_epsilon[1] =  agent1.epsilon
        # current_epsilon[2] =  agent2.epsilon
        ## Average Scores 
        scores[1] = scores[1] + ((score1 -  scores[1]) / i)
        scores[2] = scores[2] + ((score2 -  scores[2]) / i)
        pbar.set_postfix({
            "Episode" : i,
            "Agent Reward 1(O)" : score1,
            "Agent Reward 2(X)" : score2,
            "Avg Agent 1 Reward" : scores[1],
            "Avg Agent 2 Reward" : scores[2],
            "Epsilon Agent 1" : current_epsilon[1],
            "Epsilon Agent 2" : current_epsilon[2]
        })
        env.visualizeBoard()
        
        # Saving checkpoint
        if i % 1000 == 0 and i != 0:
            agent1.save_model(f"model/agent1-{i}.h5")
            # agent2.save_model(f"model/agent2-{i}.h5")
            
        pbar.update(1)
        
    pbar.close()
            
    agent1.save_model(f"model/agent-{n_game}.h5")
    # agent2.save_model(f"model/agent2-{n_game}.h5")