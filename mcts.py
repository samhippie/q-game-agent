#!/usr/bin/env python3

#see https://arxiv.org/pdf/1712.01815.pdf and the first alpha go zero
#paper for what this is based on

import copy
import random
import math
import sys
import collections
from tictactoe import TicTacToe
from checkers import Checkers
from model import MModel
import numpy as np

#generates prob table, which can be used as a policy for playing
def montecarloSearch(game, limit=100, probTable=None):
    if probTable == None:
        probTable = collections.defaultdict(lambda: (0,0))
    for i in range(limit):
        foundNewState = montecarloSearchImpl(copy.deepcopy(game), probTable)
        #if not foundNewState:
            #break
    return probTable

#picks a path according to prob table
#prob table maps (player, state, action) to (win, count)
#uct_c is the c constant used in the UCT calculation
#returns True if a new player-action-state was found
def montecarloSearchImpl(game, probTable, uct_c=1.414):
    result = None
    history = []
    while result == None:
        randomPlayout = False
        actions = game.getActions()
        player = game.turn
        state = tuple(game.getData())

        actionIndex = None
        if randomPlayout:
            #simulation
            actionIndex = random.randrange(0, len(actions))
        else:
            #selection/expansion

            #Upper Confidence Bound
            #the action with the highest UCT will be picked
            #-1 is considered to be infinity
            uctVals = []
            winCounts = []
            total = 0
            for action in actions:
                probKey = (player, state, tuple(action))
                win, count = probTable[probKey]
                winCounts.append((win, count))
                total += count

            for win, count in winCounts:
                if count != 0:
                    uct = win / count + uct_c * math.sqrt(math.log(total) / count)
                else:
                    uct = -1
                uctVals.append(uct)

            #pick the move with the highest UCT
            bestUct = None
            for i in range(len(actions)):
                uct = uctVals[i]
                #if uct is ever -1, then just pick it
                if uct == -1:
                    actionIndex = i
                    break
                elif bestUct == None or uct > bestUct:
                    bestUct = uct
                    actionIndex = i

            #start playing randomly when we expand a new node
            if bestUct == -1:
                randomPlayout = True

        #save so probTable can be updated later
        history.append((player, state, tuple(actions[actionIndex])))

        result = game.takeTurn(actions[actionIndex])

    #backpropagation

    #update probTable according to history and result
    for probKey in history:
        player, state, action = probKey
        win, count = probTable[probKey]
        if result == -1:
            probTable[probKey] = (win + 0.5, count + 1)
        elif result == player:
            probTable[probKey] = (win + 1, count + 1)
        else:
            probTable[probKey] = (win, count + 1)

    return randomPlayout

#holds the MCTS 'tree' information
#e.g. visit counts, Q values, history
class SearchState:
    def __init__(self, model, game):
        self.model = model
        self.game = game
        #cache so we don't have to touch the network as much
        self.probCache = {}
        self.valueCache = {}

        #maps (state, action) to visit count
        self.visitCount = collections.defaultdict(int)
        #list of (state, action) pairs prior to reaching leaf
        self.pendingHistory = []
        #really just used to see if something is a leaf
        self.stateVisitCount = collections.defaultdict(int)

        #average value of leaf reached for each visit
        self.qTable = collections.defaultdict(int)


    #probability vector of moves from this state
    #should we 0 out illegal actions and renorm?
    def P(self, state):
        if not state in self.probCache:
            output = self.model.getValue(state)
            prob = output['prob']
            value = output['value']
            self.probCache[state] = prob
            self.valueCache[state] = value
        return self.probCache[state]

    #expected value of the state
    def V(self, state):
        if not state in self.valueCache:
            self.P(state) # updates valueCache
        return self.valueCache[state]

    #visit count of the state-action
    def N(self, state, action):
        return self.visitCount[(state, action)]

    def addVisit(self, state, action):
        self.stateVisitCount[state] += 1
        self.visitCount[(state, action)] += 1
        self.pendingHistory.append((state, action))

    #whether a particular state has been seen before
    def isLeaf(self,state):
        return self.stateVisitCount[state] == 0

    #state is the state of the leaf node that was reached
    #the history up the that point is managed above
    def addLeafVisit(self, leafState):
        self.stateVisitCount[leafState] += 1
        #updated Q for each visit
        for state, action in self.pendingHistory:
            visits = self.N(state, action)# + 1
            #self.visitCount[(state, action)] = visits
            if visits == 1:
                #Q value is average of 1
                self.qTable[(state, action)] = self.V(leafState)
            else:
                #Q value is average of more than one
                oldQ = self.qTable[(state, action)]
                newQ = (oldQ * (visits - 1) + self.V(leafState)) / visits
                self.qTable[(state, action)] = newQ
        self.pendingHistory = []

    #part of upper bound on probability of picking state-action
    #needs a state, action, and the total list of actions from that state
    def U(self, state, action, actions):
        #no idea what this constant is supposed to be
        #sqrt(2) is just a random guess
        uConst = 1.414
        total = 0
        for b in actions:
            total += self.N(state, tuple(b))
        u = self.P(state)[self.game.enumAction(action)] * math.sqrt(total) / (1 + self.N(state, action))
        return uConst * u

    #average expected value of the state-action
    def Q(self, state, action):
        return self.qTable[(state, action)]

#generates prob table, which can be used as a policy for playing
#temperature is high for exploration, low for explotation
#(not sure what the actual value should be)
def montecarloSearchNN(model, game, searchState, limit=100,
        probTable=collections.defaultdict(lambda: (0,0)),
        temperature = 1):

    #munch on the game for a while
    for i in range(limit):
        montecarloSearchNNImpl(model, copy.deepcopy(game), searchState)

    #calculate the probabilities for the immediate actions
    actions = [tuple(a) for a in game.getActions()]
    state = tuple(game.getData())
    probVector = [0] * game.num_actions
    counts = []
    total = 0
    for action in actions:
        count = searchState.N(state, action) ** (1 / temperature)
        total += count
        counts.append(count)

    #total may be 0 depending on how SearchState is set up
    #if visit counts are updated after finding a leaf, then it can happen
    #that's not the case now, but I'm leaving the code here anyway
    if total == 0:
        print('weird corner case')
        print(counts)
        game.printBoard()
        for action in actions:
            print(game.actionToString(action))
        total = len(actions)
        counts = [1] * len(actions)

    for i in range(len(actions)):
        count = counts[i]
        prob = count / total
        action = actions[i]
        probVector[game.enumAction(action)] = prob
        probTable[(state, action)] = prob

    #return the probability vector and expected value
    #probVector will be used as a label,
    #while expected value is just from the network
    #so this is a little weird
    return (probVector, searchState.V(state))

#picks a path according to prob table and the model
#runs until it hits a new leaf node
#model is the MModel instance
#search state has all the MCTS state information
#returns True if a new player-action-state was found
def montecarloSearchNNImpl(model, game, searchState):
    result = None
    history = []
    while result == None:
        randomPlayout = False
        actions = [tuple(a) for a in game.getActions()]
        player = game.turn
        state = tuple(game.getData())

        if not searchState.isLeaf(state):
            #selection
            #pick the best Q+U
            bestAction = None
            bestQU = None
            for action in actions:
                qu = searchState.Q(state, action) + searchState.U(state, action, actions)
                if bestAction == None or qu > bestQU:
                    bestAction = action
                    bestQU = qu
            searchState.addVisit(state, bestAction)
            result = game.takeTurn(action)
        else:
            searchState.addLeafVisit(state)
            #The paper isn't clear about how to handle leaf nodes
            #just return I guess
            return True
    return False


def playWithUser(Game, probTable=collections.defaultdict(lambda:(0,0))):
    game = Game()
    userTurn = 1 if random.random() < 0.5 else 2
    print("you are player", userTurn)
    result = None
    while result == None:
        print()
        game.printBoard()
        if game.turn == userTurn:
            #show actions to user
            actions = game.getActions()
            print()
            for i in range(len(actions)):
                actionString = game.actionToString(actions[i])
                print(str(i+1) + ':', actionString)
            #perform user's action
            valid = False
            while not valid:
                userStr = input("Your Move:")
                try:
                    i = int(userStr)-1
                    if i >= 0 and i < len(actions):
                        valid = True
                except:
                    print('invalid input')
            result = game.takeTurn(actions[i])
        else:
            montecarloSearch(game, probTable=probTable, limit=1000)
            #pick the action based on probability
            actions = (game.getActions())
            data = tuple(game.getData())
            #bestAction = None
            #bestProb = None
            probs = []
            for action in actions:
                key = (game.turn, data, tuple(action))
                win, count = probTable[key]
                #count really shouldn't be 0, but let's be safe
                prob = win / max(1, count)
                print('action', action)
                print('prob', prob)
                probs.append(prob)
                #if bestAction == None or prob > bestProb:
                    #bestAction = action
                    #bestProb = prob
            probArray = np.array(probs)
            bestAction = np.random.choice(actions,
                    p=probArray / np.sum(probArray))
            result = game.takeTurn(bestAction)

    #game is over
    game.printBoard()
    print()
    print("winner:", result)

def playTrainingGame(Game, model, verbose=False):
    game = Game()
    searchState = SearchState(model, game)
    result = None
    #list of (state, player, probVector) tuples
    #state is the input, probVector is the label, player is used to determine value label
    history = []
    while result == None:
        player = game.turn
        (probVector, value) = montecarloSearchNN(model, game, searchState, limit=1000, temperature=1)
        actions = game.getActions()
        a = np.random.choice(len(probVector), p=probVector)
        if verbose:
            print('--------')
            game.printBoard(file=sys.stderr)
            for i in range(len(probVector)):
                if probVector[i] != 0:
                    print('action', game.actionToString(game.denumAction(i)), file=sys.stderr)
                    print('has probability', probVector[i], file=sys.stderr)
            print('expected value of state', value)
            print('playing', game.actionToString(game.denumAction(int(a))), file=sys.stderr)
        for action in actions:
            if game.enumAction(action) == a:
                result = game.takeTurn(action)
                break
        else:
            print('INVALID ACTION MADE IT THROUGH THE SEARCH')
            quit(1)
        history.append((game.getData(), player, probVector))

    if verbose:
        game.printBoard(file=sys.stderr)
        print('result', result, file=sys.stderr)

    #game is over, give the data to the models
    for state, player, prob in history:
        if result == -1:
            reward = 0
        elif result == player:
            reward = 1
        else:
            reward = -1
        model.addDataLabel(state, prob, reward)

def playWithUserNN(Game, model):
    game = Game()
    userTurn = 1 if random.random() < 0.5 else 2
    print("you are player", userTurn)
    result = None
    searchState = SearchState(model, game)
    while result == None:
        print()
        game.printBoard()
        if game.turn == userTurn:
            #show actions to user
            actions = game.getActions()
            print()
            for i in range(len(actions)):
                actionString = game.actionToString(actions[i])
                print(str(i+1) + ':', actionString)
            #perform user's action
            valid = False
            while not valid:
                userStr = input("Your Move:")
                try:
                    i = int(userStr)-1
                    if i >= 0 and i < len(actions):
                        valid = True
                except:
                    print('invalid input')
            result = game.takeTurn(actions[i])
        else:
            (probVector, value) = montecarloSearchNN(
                    model, game, searchState,
                    limit=100,
                    temperature=0.1)#exploitive
            print('Expected reward:', value)
            #pick the action based on probability
            actions = (game.getActions())
            a = np.random.choice(len(probVector), p=probVector)
            for action in actions:
                if game.enumAction(action) == a:
                    result = game.takeTurn(action)
                    break
            else:
                print('INVALID ACTION MADE IT THROUGH THE SEARCH')

    #game is over
    game.printBoard()
    print()
    print("winner:", result)

def train(Game, epoch_size=1000, num_epochs=200):
    #relatively high alpha for quick testing
    model = MModel(input_shape=Game.mcts_input_shape,
            num_actions=Game.num_actions,
            width=256,
            alpha=0.01)

    for i in range(num_epochs):
        print('epoch', i)
        for j in range(epoch_size):
            playTrainingGame(Game, model, verbose = j == 0)

        model.batchedUpdate(epochs=10)

    playWithUserNN(Game, model)



if __name__ == '__main__':
    train(TicTacToe, epoch_size=10, num_epochs=100)
