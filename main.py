import random


class NeuralNetwork():
    def __init__(self):
        random.seed(1)

# 2d Array representation
intialState2d = [[-1, -1, -1],
                 [0, 0, 0],
                 [1, 1, 1]]

# vector representation:
intialState = [0, -1, -1, -1, 0, 0, 0, 1, 1, 1]
termState = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]

# Helper Method that converts the string to a 2d array making it easier to parse
def convertTo2D(state):
    newstate = [[0 for x in range(3)] for x in range(3)]
    for i in range(1, len(state)):
        newstate[(i - 1) // 3][(i - 1) % 3] = state[i]

    return newstate


# Returns the player who has their current turn
# Inputs:
# s: vector representation of the state of the game
# Returns:
# "white": if the first value of the array is 0 indicating it is white's turn
# "black": if the first value of the array is 1 indicating it is black's turn
def toMove(s):
    if s[0] == 0:
        return "white"
    if s[0] == 1:
        return "black"
    return "ERROR"

# Helper function that determines the game is in a terminal state for black
def terminalBlack(s):
    for i in range(1, len(s)):
        print(i)
        print(s[i])
        if s[i] == -1:
            if (i +3 <= 9 and s[i+3] == 0):
                return False
            if (i + 2 <= 1 and s[i+2] == 1):
                return False
            if(i + 4 <= 9 and s[i+4] == 1):
                return False
    return True

# Helper function that determines the game is in a terminal state for white
def terminalWhite(s):
    for i in range(1, len(s)):
        print(i)
        print(s[i])
        if s[i] == 1:
            if (i -3 >= 1 and s[i-3] == 0):
                return False
            if (i - 2 >= 1 and s[i-2] == -1):
                return False
            if(i - 4 >= 1 and s[i-4] == -1):
                return False
    return True
    """
    #My shame
    for i in range(len(s)):
        for j in range(len(s[0])):
            if s[i][j] == 1:
                print(s[i][j])
                print(i,j)
                print(s[i-1][j])
                if (i-1 < 0 and s[i-1][j] == 0) or (i-1 < 0 and j-1 < 0 and s[i-1][j-1] == 1) or (i-1 < 0 and j+1 > 2 and s[i-1][j+1] == -1):
                    return False
    return True
    """
# Determines if the game can be continued or not
# Inputs:
# s: the current game state
# Returns:
# True or False: based on if the game is a at a terminal state
def isTerminal(s):
    return terminalWhite(s) and terminalBlack(s)


def actions(s):
    legalMoves = []
    if isTerminal(s):
        return None
    if toMove(s) == "white":
        for i in range(1, len(s)):
            if (i -3 >= 1 and s[i-3] == 0):
                legalMoves.append("advance", i//3, i%3)
            if (i - 2 >= 1 and s[i-2] == -1):
                legalMoves.append("attack-right", i//3, i%3)
            if(i - 4 >= 1 and s[i-4] == -1):
                return legalMoves.append("attack-left", i//3, i%3)
        return legalMoves
    if toMove(s) == "black":
        return None

    # Illegal to get here
    return "ERROR"


def result(s, a):
    return None


def utility(s):
    return None


def minimaxsearch():
    return None


def classify():
    return None


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Hello World")
    #print(convertTo2D(intialState))
    print(isTerminal(intialState))
    print(isTerminal(termState))
    #print()
