# ref: https://www.youtube.com/watch?v=5fHngyN8Qhw

from tensorflow.keras.layers import Dense, Activation, Dropout, Embedding, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


###############################################################################
class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.9999,  epsilon_end=0.1,
                 mem_size=1000000, fname='ddqn_model.h5', replace_target=100, load_model=None):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions,
                                   discrete=True)
        self.q_eval = self.build_dqn(alpha, n_actions, input_dims)
        self.q_target = self.build_dqn(alpha, n_actions, input_dims)
        if load_model is not None:
            self.q_eval = tf.keras.models.load_model(load_model)
            self.q_target = tf.keras.models.load_model(load_model)
            
    def build_dqn(self, lr, n_actions, input_dims):
        model = Sequential()
        model.add(Embedding(3, 3, input_shape=(input_dims,)))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(n_actions, activation='linear'))
        
        model.compile(optimizer=Adam(lr=lr), loss='mse')
        
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def available_move(self,arr) -> np.array:
        fin = []
        for i in range(len(arr)):
            if arr[i] == 0:
                fin.append(i)
        return np.array(fin)

    def choose_action(self, state):
        valid_move = self.available_move(state)
        state = np.array(state)
        state = state[np.newaxis, :]
        rand = np.random.random()
        action = np.random.choice(valid_move)
        if rand >= self.epsilon:
            actions = self.q_eval.predict(state)[0]
            pred = np.argsort(actions, kind='heapsort')[::-1]
            for i in pred:
                if i in valid_move:
                    action = i
                    return i
        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                          self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)
            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + \
                    self.gamma*q_next[batch_index, max_actions.astype(int)]*done

            _ = self.q_eval.fit(state, q_target, verbose=0)

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                           self.epsilon_min else self.epsilon_min
            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()
        return

    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def save_model(self, name=None):
        if name == None:
            name = self.model_file
        self.q_eval.save(name)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        # if we are in evaluation mode we want to use the best weights for
        # q_target
        if self.epsilon == 0.0:
            self.update_network_parameters()