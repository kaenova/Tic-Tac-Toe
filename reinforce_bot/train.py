#! python

# Disable tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()
tf.get_logger().setLevel("CRITICAL")
tf.autograph.set_verbosity(0)


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)
# exit(0)

from agent import DDQNAgent
from game_env import Game
import time
from tqdm import tqdm


if __name__=='__main__':
    # Initialize Game, number of game, and the agent to against eachother
    env = Game()
    n_game = 10000
    agent1 = DDQNAgent(alpha=0.005, gamma=1, 
                       epsilon=1, epsilon_end=0.05, epsilon_dec=0.999,
                       input_dims=env.num_state, n_actions=env.num_action,
                       batch_size=32) # Agen mengisi O
    agent2 = DDQNAgent(alpha=0.005, gamma=1, 
                       epsilon=1,  epsilon_end=0.05, epsilon_dec=0.999,
                        input_dims=env.num_state, n_actions=env.num_action,
                        batch_size=32) # Agen mengisi X
    
    # Initialize agent performance logging
    scores = {
        1: 0,
        2: 0
    }
    current_epsilon = {
        1: 1,
        2: 1
    }
    
    # Training part
    pbar = tqdm(total=n_game)
    try:
        for i in range(1, n_game):
            start = time.time()
            ### Initialize Scoring
            done = False
            score1 = 0
            score2 = 0
            action1 = 0
            action2 = 0
            observation = env.reset()
            while not done:
                # Agent Turn
                if env.round % 2 == 0:
                    action1 = agent1.choose_action(observation)
                    observation_, reward, done = env.step(action1)
                    score1 += reward
                    if env.player_won == 1:
                        score2 += -10
                    if env.player_won == 0: # Check on final turn if its draw
                        score2 += -score2
                        score1 += -score1
                else:
                    action2 = agent2.choose_action(observation)
                    observation_, reward, done = env.step(action2)
                    score2 += reward
                    if env.player_won == 2:
                        score1 += -10
                        
                if env.round % 2 == 0 or env.player_won != -1:
                    agent1.remember(observation, action1, score1, observation_, done)
                if env.round % 2 == 1 or env.player_won != -1:
                    agent2.remember(observation, action2, score2, observation_, done)
                    
                observation = observation_
            
            agent1.learn()
            agent2.learn()
            
            # Logging
            current_epsilon[1] =  agent1.epsilon
            current_epsilon[2] =  agent2.epsilon
            ## Average Scores 
            scores[1] = scores[1] + ((score1 -  scores[1]) / i)
            scores[2] = scores[2] + ((score2 -  scores[2]) / i)
            pbar.set_postfix({
                "Episode" : i,
                "Agent 1 Score" : score1,
                "Agent2 Score" : score2,
                "Avg Agent 1 Reward" : scores[1],
                "Avg Agent 2 Reward" : scores[2],
                "Epsilon Agent 1" : current_epsilon[1],
                "Epsilon Agent 2" : current_epsilon[2],
            })
            print(env)

            # Saving checkpoint
            if i % 1000 == 0 and i != 0:
                agent1.save_model(f"model/agent-{i}.h5")
                # agent2.save_model(f"model/agent2-{i}.h5")
                
            pbar.update(1)
            
    except KeyboardInterrupt:
        pass
    
    pbar.close()
    path1 = f"model/agent1-{i}.h5"
    path2 = f"model/agent2-{i}.h5"
    print(f"Saving agent to {path1} and {path2}")
    agent1.save_model(path1)
    agent2.save_model(path2)
    exit(0)