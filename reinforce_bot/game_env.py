# A class for agent interaction with the game
from typing import List
import numpy as np

class Game:
  
    def __init__(self) -> None:
        self.num_action = 9
        self.num_state = 9
        self.player1_label = 1
        self.player2_label = 2
        self.reset()
    
    def reset(self) -> List[int]:
        self.board = [0,0,0,0,0,0,0,0,0]
        self.round = 0
        self.done = False
        self.player_won = -1 # -1 no one wins (on going game), 0 draw, 1 player 1 wins, 2 player 2 wins
        return self.create_state()
    
    def create_state(self) -> List[int]:
        arr = self.board[:]
        return arr
        
    def available_move(self):
        arr = []
        for i in range(len(self.board)):
            if self.board[i] == 0:
                arr.append(i)
        return arr
        
    def step(self, idx: List[int]):
        # Not valid moves! Get minus -100 rewards
        if idx not in self.available_move():
            rewards = -100
            board_state = self.create_state()
            return board_state, rewards, False
        
        # Assigning simbols with an index and check
        if self.round % 2 == 0: # Player 1 input
            self.board[idx] = self.player1_label
        else:                   # Player 2 input
            self.board[idx] = self.player2_label
            
        player_won = self.check_won()
        if player_won != -1:
            self.set_game_end(player_won)
            
        # Calculating reward
        reward = self.__calculate_rewards()
        
        # Add round
        self.round += 1
        return self.create_state(), reward, self.done

    def __calculate_rewards(self):
        reward = 1
        # Game Draw
        if self.player_won == 1 or self.player_won == 2:
            reward = 10
        return reward
    
    def check_won(self) -> int:
        if self.player_one_won():
            return 1
        if self.player_two_won():
            return 2
        if self.is_draw():
            return 0
        return -1
    
    def player_one_won(self) -> bool:
        if (
               ((self.board[0] == 1) and (self.board[1] == 1) and (self.board[2] == 1))
            or ((self.board[0] == 1) and (self.board[4] == 1) and (self.board[8] == 1))
            or ((self.board[0] == 1) and (self.board[3] == 1) and (self.board[6] == 1))
            or ((self.board[1] == 1) and (self.board[4] == 1) and (self.board[7] == 1))
            or ((self.board[2] == 1) and (self.board[5] == 1) and (self.board[8] == 1))
            or ((self.board[2] == 1) and (self.board[4] == 1) and (self.board[6] == 1))
            or ((self.board[3] == 1) and (self.board[4] == 1) and (self.board[5] == 1))
            or ((self.board[6] == 1) and (self.board[7] == 1) and (self.board[8] == 1))
        ):
            return True
        else:
            return False
        
    def player_two_won(self) -> bool:
        if (
            ((self.board[0] == 2) and (self.board[1] == 2) and (self.board[2] == 2))
            or ((self.board[0] == 2) and (self.board[4] == 2) and (self.board[8] == 2))
            or ((self.board[0] == 2) and (self.board[3] == 2) and (self.board[6] == 2))
            or ((self.board[1] == 2) and (self.board[4] == 2) and (self.board[7] == 2))
            or ((self.board[2] == 2) and (self.board[5] == 2) and (self.board[8] == 2))
            or ((self.board[2] == 2) and (self.board[4] == 2) and (self.board[6] == 2))
            or ((self.board[3] == 2) and (self.board[4] == 2) and (self.board[5] == 2))
            or ((self.board[6] == 2) and (self.board[7] == 2) and (self.board[8] == 2))
        ):
            return True
        else:
            return False
        
    def is_draw(self) -> bool:
        arr = self.board[:]
        arr = np.array(arr)
        return not ((arr == 0).any())
    
    def set_game_end(self, player_num_won: int):
        """
        1 for player 1 wins,
        2 for player 2 wins,
        0 for a draw
        """
        self.player_won = player_num_won
        self.done = True
    
    def __repr__(self) -> str:
        """
        // Box (Board) State
        box = 0 -> empty
        box = 1 -> O
        box = 2 -> X
        """
        board = self.board
        translatedBoard = []
        for box in board:
            if box == 0:
                translatedBoard.append(" ")
            elif box == 1:
                translatedBoard.append("O")
            elif box == 2:
                translatedBoard.append("X")
            else:
                print("Something is wrong here.")
                
        return f"""
    \t Round {self.round} Player Won {self.player_won}
    \t ---+---+---
    \t  {translatedBoard[0]} | {translatedBoard[1]} | {translatedBoard[2]} 
    \t ---+---+---
    \t  {translatedBoard[3]} | {translatedBoard[4]} | {translatedBoard[5]} 
    \t ---+---+---
    \t  {translatedBoard[6]} | {translatedBoard[7]} | {translatedBoard[8]} 
    \t ---+---+---
    """