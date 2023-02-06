from copy import deepcopy
from tensorflow.keras.models import load_model
import numpy as np

class ReinforceBot:
    
    EPSILON = 0.05
    
    def __init__(self, player_num : int, path: str):
        self.q_eval = None
        self.p_number = player_num
        self.load_model(path)
        
    def action(self, board) -> int:
        board_copy = board[:]
        valid_move = self._available_move(board_copy)
        state = np.array(board_copy)
        state = state[np.newaxis, :]
        rand = np.random.random()
        action = np.random.choice(valid_move)
        if rand >= self.EPSILON:
            actions = self.q_eval.predict(state)[0]
            pred = np.argsort(actions, kind='heapsort')[::-1]
            for i in pred:
                if i in valid_move:
                    return i
        return action
    
    def _available_move(self,board) -> np.array:
        fin = []
        for i in range(len(board)):
            if board[i] == 0:
                fin.append(i)
        
        return np.array(fin)
    
    def load_model(self, path):
        self.q_eval = load_model(path)