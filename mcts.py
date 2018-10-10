#!/usr/bin/env python3

import copy
import random
import math
import collections
from tictactoe import TicTacToe
from checkers import Checkers

#generates prob table, which can be used as a policy for playing
def markovSearch(game, limit=100, probTable=None):
    if probTable == None:
        probTable = collections.defaultdict(lambda: (0,0))
    for i in range(limit):
        foundNewState = markovSearchImpl(copy.deepcopy(game), probTable)
        #if not foundNewState:
            #break
    return probTable

#picks a path according to prob table
#prob table maps (player, state, action) to (win, count)
#uct_c is the c constant used in the UCT calculation
#returns True if a new player-action-state was found
def markovSearchImpl(game, probTable, uct_c=1.414):
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
            markovSearch(game, probTable=probTable, limit=1000)
            #pick the best action
            actions = (game.getActions())
            data = tuple(game.getData())
            bestAction = None
            bestProb = None
            for action in actions:
                key = (game.turn, data, tuple(action))
                win, count = probTable[key]
                #count really shouldn't be 0, but let's be safe
                prob = win / max(1, count)
                print('action', action)
                print('prob', prob)
                if bestAction == None or prob > bestProb:
                    bestAction = action
                    bestProb = prob
            result = game.takeTurn(bestAction)

    #game is over
    game.printBoard()
    print()
    print("winner:", result)

if __name__ == '__main__':
    playWithUser(Checkers)
