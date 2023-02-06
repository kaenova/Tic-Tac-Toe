import os
from reinforce_bot.bot_interface import ReinforceBot
import random

def visualizeBoard(board):
    """
    This function is used to print out the board state on console.
    // Box (Board) State
    box = 0 -> empty
    box = 1 -> O
    box = 2 -> X
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print('=========== BOARD ===========\n')
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

    print(f"\t  {translatedBoard[0]} | {translatedBoard[1]} | {translatedBoard[2]} ")
    print("\t ---+---+---")
    print(f"\t  {translatedBoard[3]} | {translatedBoard[4]} | {translatedBoard[5]} ")
    print("\t ---+---+---")
    print(f"\t  {translatedBoard[6]} | {translatedBoard[7]} | {translatedBoard[8]} ")

def nextTurn(board, round):
    """
    This function is used for player input. If it's on odd round then the player "1",
    if its even then the player "2"
    // Box (Board) State
    box = 0 -> empty
    box = 1 -> O
    box = 2 -> X
    """
    endTurn = 0
    while endTurn == 0 :
        visualizeBoard(board)
        print(f'\n======= PLAYER {int(round % 2 == 1)+1} MOVE =======\n')
        print(f'List of empty position:')

        listPos = []
        order = 0
        for i in range(len(board)):
            if board[i] == 0:
                print(f"{order+1}. Position {i+1}")
                listPos.append(i)
                order += 1
        
        move = int(input('\nYour move => ')) - 1

        if move < len(listPos):
            if (int(round % 2 == 1)+1) == 1: # Player 1 got to play
                board[listPos[move]] = 1
            elif (int(round % 2 == 1)+1) == 2: # Player 2 got to play
                board[listPos[move]] = 2
            endTurn = 1

    return board

def ruleChecker(board):
    """
    This function is used to check if there's a valid winner
    """
    visualizeBoard(board)
    if (
        ((board[0] == 1) and (board[1] == 1) and (board[2] == 1))
        or ((board[0] == 2) and (board[1] == 2) and (board[2] == 2))
        or ((board[0] == 1) and (board[4] == 1) and (board[8] == 1))
        or ((board[0] == 2) and (board[4] == 2) and (board[8] == 2))
        or ((board[0] == 1) and (board[3] == 1) and (board[6] == 1))
        or ((board[0] == 2) and (board[3] == 2) and (board[6] == 2))
        or ((board[1] == 1) and (board[4] == 1) and (board[7] == 1))
        or ((board[1] == 2) and (board[4] == 2) and (board[7] == 2))
        or ((board[2] == 1) and (board[5] == 1) and (board[8] == 1))
        or ((board[2] == 2) and (board[5] == 2) and (board[8] == 2))
        or ((board[2] == 1) and (board[4] == 1) and (board[6] == 1))
        or ((board[2] == 2) and (board[4] == 2) and (board[6] == 2))
        or ((board[3] == 1) and (board[4] == 1) and (board[5] == 1))
        or ((board[3] == 2) and (board[4] == 2) and (board[5] == 2))
        or ((board[6] == 1) and (board[7] == 1) and (board[8] == 1))
        or ((board[6] == 2) and (board[7] == 2) and (board[8] == 2))
    ):
        return 1
    else:
        return 0

def gamePlay2Players():
    """
    Main Game function.
    """
    board = [0,0,0,0,0,0,0,0,0]
    winStatus = 0
    round = 0
    while (round < 9) and (winStatus == 0):
        board = nextTurn(board, round)
        winStatus = ruleChecker(board)
        input('\nPress enter to continue => ')
        if winStatus == 0:
            round += 1
            if round == 9:
                print("No one wins!")
        else:
            print(f'Player {int(round % 2 == 1)+1} won the game!')
            
def gamePlayBot(bot_player: int):
    """
    Main Game Function with BOT
    """
    bot_p1_agent_path = "reinforce_bot/model/agent1-3613.h5"
    bot_p2_agent_path = "reinforce_bot/model/agent2-3613.h5"
    board = [0,0,0,0,0,0,0,0,0]
    winStatus = 0
    round = 0
    bot_agent_path = bot_p1_agent_path if bot_player == 1 else bot_p2_agent_path
    botAgent = ReinforceBot(bot_player, bot_agent_path)
    while (round < 9) and (winStatus == 0):
        if (round % 2) == (bot_player - 1):
            if round == 0:
                idx = random.randint(0, 8)
            else:
                idx = botAgent.action(board)
            board[idx] = bot_player
        else:
            board = nextTurn(board, round)
            
        winStatus = ruleChecker(board)
        input('\nPress enter to continue => ')
        if winStatus == 0:
            round += 1
            if round == 9:
                print("No one wins!")
        else:
            print(f'Player {int(round % 2 == 1)+1} won the game!')
    
            
def botMenu():
    os.system('cls' if os.name == 'nt' else 'clear')
    print('=========== REINFORCEMENT BOT MENU ===========\n')
    print('1. Bot play first')
    print('2. Bot play second')
    print('3. Back\n')
    optSelection = input("Select your input => ")
    if int(optSelection) < 3 and int(optSelection) > 0:
        gamePlayBot(int(optSelection))


def mainMenu():
    exitStatus = 0
    while exitStatus == 0:
        # os.system('cls' if os.name == 'nt' else 'clear')
        print('=========== MAIN MENU ===========\n')
        print('1. Start the game')
        print('2. Play with reinfoce bot')
        print('3. Exit the game\n')
        optSelection = input("Select your input => ")
        if optSelection == "1":
            gamePlay2Players()
            exitStatus = 1
        elif optSelection == "2":
            botMenu()
        elif optSelection == "3":
            exitStatus = 1