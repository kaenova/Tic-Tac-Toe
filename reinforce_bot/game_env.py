# A class for agent interaction with the game
from copy import deepcopy

class Game:
    board = [0,0,0,0,0,0,0,0,0]
    num_action = 9
    num_state = 10
    
    
    player1_label = 1
    player2_label = 2
    
    round = 0
    done = False
    
    def __init__(self):
        pass
    
    def create_state(self):
        arr = deepcopy(self.board)
        # Buat state untuk nge save dia pemain 1 atau 2
        if self.round % 2 == 0:
            player_num = 1
        else:
            player_num = 2
        arr.append(player_num)
        
        return arr
    
    def reset(self):
        self.board = [0,0,0,0,0,0,0,0,0]
        self.round = 0
        return self.create_state()
        
    def rule_check(self):
        if (
               ((self.board[0] == 1) and (self.board[1] == 1) and (self.board[2] == 1))
            or ((self.board[0] == 2) and (self.board[1] == 2) and (self.board[2] == 2))
            or ((self.board[0] == 1) and (self.board[4] == 1) and (self.board[8] == 1))
            or ((self.board[0] == 2) and (self.board[4] == 2) and (self.board[8] == 2))
            or ((self.board[0] == 1) and (self.board[3] == 1) and (self.board[6] == 1))
            or ((self.board[0] == 2) and (self.board[3] == 2) and (self.board[6] == 2))
            or ((self.board[1] == 1) and (self.board[4] == 1) and (self.board[7] == 1))
            or ((self.board[1] == 2) and (self.board[4] == 2) and (self.board[7] == 2))
            or ((self.board[2] == 1) and (self.board[5] == 1) and (self.board[8] == 1))
            or ((self.board[2] == 2) and (self.board[5] == 2) and (self.board[8] == 2))
            or ((self.board[2] == 1) and (self.board[4] == 1) and (self.board[6] == 1))
            or ((self.board[2] == 2) and (self.board[4] == 2) and (self.board[6] == 2))
            or ((self.board[3] == 1) and (self.board[4] == 1) and (self.board[5] == 1))
            or ((self.board[3] == 2) and (self.board[4] == 2) and (self.board[5] == 2))
            or ((self.board[6] == 1) and (self.board[7] == 1) and (self.board[8] == 1))
            or ((self.board[6] == 2) and (self.board[7] == 2) and (self.board[8] == 2))
        ):
            return 1
        else:
            return 0
        
    def available_move(self):
        arr = []
        for i in range(len(self.board)):
            if self.board[i] == 0:
                arr.append(i)
        
        return arr
        
    def step(self, idx: int):
        done = False
        # Not valid moves! Get minus -20 rewards
        if idx not in self.available_move():
            return self.create_state(), -20, True
        
        # Assigning simbols with an index
        if self.round % 2 == 0: # Player 1 input
            self.board[idx] = self.player1_label
        else:                   # Player 2 input
            self.board[idx] = self.player2_label
        
        self.round += 1
        reward = -1
        if self.rule_check():
            done = True
            reward = 10
            return self.create_state(), reward, done
            
        if self.round >= 8:
            reward = 5
            if self.round == 9: done = True
            
        return self.create_state(), reward, done
    
    def latestBoardStr(self) -> str:
        """
        This function is used to print out the board state on console.
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
    =========== END BOARD ===========
    \t  {translatedBoard[0]} | {translatedBoard[1]} | {translatedBoard[2]} 
    \t ---+---+---
    \t  {translatedBoard[3]} | {translatedBoard[4]} | {translatedBoard[5]} 
    \t ---+---+---
    \t  {translatedBoard[6]} | {translatedBoard[7]} | {translatedBoard[8]} 
    """
    
    def visualizeBoard(self) -> str:
        print(self.latestBoardStr())