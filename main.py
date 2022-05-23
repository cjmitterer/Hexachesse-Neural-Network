import random
import numpy as np

# 2d Array representation
intialState2d = [[-1, -1, -1],
                 [0, 0, 0],
                 [1, 1, 1]]

# vector representation:
intialState = [0, -1, -1, -1, 0, 0, 0, 1, 1, 1]
termState = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]

policyTable = {}
"""PART 3:"""
# Neural Network defined as a class
# contains predict method, sigmoid, relu, and updateWeights
# Values of the self are learning rate bias and weights
# And weight are seeded to preserve values
class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.seed(1), np.random.seed(2)])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    # sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # relu function
    def relu(self, x):
        max(0, x)

    # sigmoid derivative
    def sigmoidDeriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def reluDeriv(self, x):
        if x < 0:
           return 0
        else:
            return 1

    def predict(self, s):
        layerOne = np.dot(s, self.weights) + self.bias
        layerTwo = np.dot(layerOne, self.weights) + self.bias
        layerThreeSigmoid = self.sigmoid(layerTwo)
        layerThreeRelu = self.relu(layerTwo)

        prediction = layerThreeSigmoid
        return prediction

    def updateWeights(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
                derror_dweights * self.learning_rate
        )


# Helper Method that converts the string to a 2d array making it easier to parse
def convertTo2D(state):
    newstate = [[0 for x in range(3)] for x in range(3)]
    for i in range(1, len(state)):
        newstate[(i - 1) // 3][(i - 1) % 3] = state[i]

    return newstate

"""
PART 1:
"""
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
    """
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
    return True

# Determines if the game can be continued or not
# Calls the black and white terminal checkers to see if there exists such state for either
# Inputs:
# s: the current game state
# Returns:
# True or False: based on if the game is a at a terminal state
def isTerminal(s):
    return terminalWhite(s) and terminalBlack(s)


# Takes the current state of the board and returns a dictionary of all possible moves
# that can be executed
# Inputs:
# s: the current game state
# Returns:
# A dictionary of the possible moves that can be made with a key being the current board state
def actions(s):
    legalMoves = {}                                             # Uses dictionary for faster runtime
    if isTerminal(s):
        return None
    if toMove(s) == "white":
        for i in range(1, len(s)):
            if(s[i] == 1):
                if (i -3 >= 1 and s[i-3] == 0):
                    legalMoves[s].append("advance", i//3, i%3)
                if (i - 2 >= 1 and s[i-2] == -1):
                    legalMoves[s].append("attack-right", i//3, i%3)        # Fix the logic later but its "fine" now
                if(i - 4 >= 1 and s[i-4] == -1):
                    legalMoves[s].append("attack-left", i//3, i%3)         # Fix the logic later but its "fine" now
        return legalMoves
    if toMove(s) == "black":
        for i in range(1, len(s)):
            if(s[i] == -1):
                if (i + 3 <= 9 and s[i+3] == 0):
                    legalMoves[s].append("advance", i//3, i%3)
                if (i + 2 <= 9 and s[i+2] == 1):
                    legalMoves[s].append("attack-left", i//3, i%3)         # Fix the logic later but its "fine" now
                if (i + 4 <= 9 and s[i+4] == 1):
                    legalMoves[s].append("attack-right", i//3, i%3)        # Fix the logic later but its "fine" now
    return legalMoves

# Takes a state an action and returns a new state with the action executed
# Input:
# s (sting): The current state of the board
# a (string): A string input of what action to take the action and what where the piece is currently
# Return:
# newState (string): The new current board state with the action, a, executed on s
def result(s, a):
    newState = s
    listActions = a.spit()      # [0] is the action [1] is the row [2] is the column
    newState[3*a[1] + a[2] + 1] = 0         # Sets the old position to 0. 3*times row + column + 1
    if(toMove(s) == "white"):
        if listActions[0] == "advance":
            newState[(3*int(a[1]) + int(a[2]) + 1) - 3] = 1   # I can just know that this is acceptable because only valid inputs are given
        elif listActions[0] == "attack-right":
            newState[(3*int(a[1]) + int(a[2]) + 1) - 2] = 1
        elif listActions[0] == "attack-left":
            newState[(3*int(a[1]) + int(a[2]) + 1) - 4] = 1

    elif(toMove(s) == "black"):
        if listActions[0] == "advance":
            newState[(3*int(a[1]) + int(a[2]) + 1) + 3] = -1
        elif listActions[0] == "attack-left":
            newState[(3*int(a[1]) + int(a[2]) + 1) + 2] = -1
        elif listActions[0] == "attack-right":
            newState[(3*int(a[1]) + int(a[2]) + 1) + 4] = 1
    newState[0] = (int(newState[0]) + 1) % 2                # Switches who's turn its up
    return newState


# Have to assume the utility is called only when isTerminal otherwise
# A function that allots points based on whom won
# Inputs:
# s (string): state of the board
# p (string): Who's turn the player is
# Returns:
# 1 if white wins and 0 if black wins
# None is "possible but that should never be reached
def utility(s, p):
    if p == "white" and isTerminal(s):
        return 1
    elif p == "black" and isTerminal(s):
        return 0
    # Shouldn't be reached
    else:
        return None


"""
PART 2:
"""
# Used to build the policy table for hexapawn
# Called recursively that either calls utility if terminal, minimax based on results for white and then black
# Inputs:
# s (string): the current state of the board
# Outputs:
# Returns:
# The policy for the hexapawn round and points alloted
def minimaxsearch(s):
    if isTerminal(s):
        return utility(s, toMove(s))
    elif toMove(s) == "white":
        for i in range(actions(s)[s]):
            return minimaxsearch(result(s, i))
    elif toMove(s) == "black":
        for i in range(actions(s)[s]):
            return minimaxsearch(result(s, i))

# Generates list of actions executed
# Effects:
# Modifies the policy table even outside the function since it is a dictionary
def createPolicyTable(s):
    for i in range(actions(s)[s]):
        policyTable.append(actions(s)[s])
    return policyTable

"""
Part 4:
"""
# Inputs:
# neuralNetwork: the neural network being used to train
# s: the state of the board
# Returns:
# calls the predict function from the defined neuralNetwork class which already
# classifies it
# neuralNetwork: The created neural network class
# s: the board state
# Results:
# Prediction vector based off of the the predict neural network command
def classify(neuralNetwork, s):
    return neuralNetwork.predict(minimaxsearch(s))



"""
Part 5:
"""
# instance of the network you designed in vector of expected outputs
# Uses back propagation to modify the weights
# in your network based on the differences between the expected outputs
# Inputs:
# neuralNetwork: The created neural network class
# s: the board state
# Returns:
# newWeighting:
# Calls the updateWeight in the neuralNetwork class and loops through it based on the provided
# inputs
def updateWeights(neuralNetwork, s):
    newWeighting = neuralNetwork.updateWeights(classify(neuralNetwork, s))
    for i in range(newWeighting):
        newWeighting = neuralNetwork.updateWeights(classify(newWeighting, s))
    return newWeighting

if __name__ == '__main__':
    print("Hello World")
    #print(convertTo2D(intialState))
    #print(isTerminal(intialState))
    #print(isTerminal(termState))
    #print()

"""
PART: 6

Limited amount of neural networks to reduce overfitting
3-4 layers and limited amount of neurons ~2-3
When using a seeded random value for input it's runtime to train it is not that long
I lack the data but for such a simple problem there are is
limited amount of layers needed at risk of overfitting.

For this type of problem a relu is usually preferable over sigmoid as relu simply checks the max
value and sees if its wanted
"""
