#!/usr/bin/env python3

import copy
import random
from model import QModel

discount = 0.99
epsilon = 0.3

#manages consecutive states for a given model
#this requires that updateState() will be called before getAction() for a particular turn
class StateManager:
    def __init__(self, model, modelTuples, verbose=False):
        self.model = model
        self.state = None
        self.action = None
        self.modelTuples = modelTuples
        self.verbose = verbose

    #gets the action for the currently saved state
    def getAction(self):
        if self.state == None:
            raise Error('invalid state')
        self.action = getAction(self.model, self.state, self.verbose)
        return self.action

    #saves the new state, updating the model using previous state,
    #previous action, and new state
    def updateState(self, newState, reward=0):
        #we can't let the saved state be mutated
        newStateCopy = copy.deepcopy(newState)
        if self.state != None and self.action != None:
            self.modelTuples.append((self.action, self.state, newStateCopy, reward))
            #updateAction(self.model, self.action, self.state, newState, reward)
        self.state = newStateCopy

#uses both models to play through a game,
#updating both as it goes
def playGame(game, model1, model2, modelTuples1, modelTuples2, verbose=False):
    #have a separate manager for states
    #which will work for games that don't just alternate turns
    mans = [StateManager(model1, modelTuples1, verbose), StateManager(model2, modelTuples2, verbose)]
    result = None
    while result == None:
        if(verbose):
            print()
            game.printBoard()
        man = mans[game.turn - 1]
        man.updateState(game)
        action = man.getAction()
        result = game.takeTurn(action)

    #tie, no reward
    if result == -1:
        mans[0].updateState(None)
        mans[1].updateState(None)
    #win/loss, +/- reward
    else:
        mans[0].updateState(None, reward= 1 if result == 1 else -1)
        mans[1].updateState(None, reward=-1 if result == 1 else 1)

    if(verbose):
        print()
        game.printBoard()
    return game

#uses the model to play a game with the user
#does not update the model
def playWithUser(Game, model):
    game = Game()
    userTurn = 1 if random.random() > 0.5 else 2
    print("you are player", userTurn)
    while True:
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
            i = int(input("Your Move:"))-1
            result = game.takeTurn(actions[i])
        else:
            #get the action from the model and perform it
            action = getAction(model, game)
            result = game.takeTurn(action)

        if result:
            game.printBoard()
            print()
            print("winner:", result)
            return game

#gets the best action for the given state according to the given model
def getBestAction(model, state, verbose=False):
    #simply pick the action with the highest expected reward
    bestAction = None
    bestValue = None

    #get actions values in a batch
    actions = state.getActions()
    stateData = state.getData()
    actionInputs = []
    for action in actions:
        actionInputs.append(stateData + action)
    actionValues = model.getValues(actionInputs)

    #find the highest value action
    for i in range(len(actionValues)):
        if(verbose):
            print('input', actions[i])
            print('value', actionValues[i])
        if bestAction == None or actionValues[i] > bestValue:
            bestValue = actionValues[i]
            bestAction = actions[i]

    return bestAction

#gets an action
#random epsilon * 100% of the time
#best (1-epsilon) * 100% of the time
def getAction(model, state, verbose=False):
    if epsilon == 1 or random.random() < epsilon:
        return random.choice(state.getActions())
    else:
        return getBestAction(model, state, verbose)

#updates the value of the action in the given state,
#given that it led to nextState
def updateAction(model, action, state, nextState, reward):
    val = discount * getStateValue(model, nextState) + reward
    model.updateTable(state.getData() + action, val)

#gets the value of the given state
#which is defined as the value of its most valuable action
#if defined, alt_model is used to select the best action
#if not defined, model will be used
#either way, model is used for the final evaluation
def getStateValue(model, state, sort_model=None):
    if sort_model == None:
        sort_model = model
    #sometimes there isn't a next state
    #0 means that terminal state values are purely determined by reward
    if state == None:
        return 0

    #get actions values in a batch
    actions = state.getActions()
    stateData = state.getData()
    actionInputs = []
    for action in actions:
        actionInputs.append(stateData + action)
    actionValues = sort_model.getValues(actionInputs)

    #find the highest value action
    bestAction = None
    bestValue = None
    for i in range(len(actionValues)):
        if bestAction == None or actionValues[i] > bestValue:
            bestValue = actionValues[i]
            bestAction = actions[i]

    #return the value of the best action
    if bestValue == None:
        return 0
    else:
        return model.getValue(stateData + bestAction)


#makes/loads some models and trains them according to the parameters
#then lets the user play one of the models
#epoch size is the number of games played between training sessions
#num epochs is the number of epochs before finishing
#random epochs is the number of initial epochs where epsilon = 1 i.e. complete random
#sample size is the number of tuples evaluated at a time in a training session
#num samples is the number of samples used to train in a training session
#target clone steps is how many steps before the target network is cloned from the main network
#saveDir should not have a trailing / (unless you're using the filesystem root)
def train(Game, name, num_models=3, epoch_size=1000, num_epochs=200, random_epochs=0, sample_size=100, target_clone_steps=5, num_samples=100, saveDir='.', loadModels=False):
    #init models
    models = []
    targetModels = []
    modelTuples = []
    for i in range(num_models):
        model = QModel(input_shape=Game.input_shape, alpha=0.001)
        if loadModels:
            filename = saveDir + '/' + 'model-' + name + str(i) + '.h5'
            try:
                model.loadModel(filename)
            except:
                print('failed to load', filename)
        models.append(model)
        targetModels.append(model.clone())
        modelTuples.append([])

    global epsilon

    if num_epochs == 0:
        #just show a sample game, don't do any training
        oldEpsilon = epsilon
        epsilon = 0
        game = playGame(Game(), models[0], models[-1], [], [], verbose=True)
        print('-----------------')
        epsilon = oldEpsilon

    #save epsilon so we can have a few random epochs
    nonRandomEpsilon = epsilon
    epsilon = 1

    for i in range(num_epochs):
        #show a sample game each epoch
        oldEpsilon = epsilon
        epsilon = 0
        game = playGame(Game(), models[0], models[-1], [], [], verbose=True)
        print('-----------------')
        epsilon = oldEpsilon
        print(i)

        #restore epsilon after the random epochs are over
        if i == random_epochs:
            epsilon = nonRandomEpsilon

        #every so often, update the target models and clear out the stored tuples
        #these don't have to happen at the same time, but this is convenient
        if i % target_clone_steps == 0:
            for j in range(len(models)):
                targetModels[j] = models[j].clone()
                modelTuples[j] = []

        #epoch
        print('playing')
        for j in range(epoch_size):
            #this way of sampling means that every model plays every model once
            #before playing the same model again (assuming n*n divides epoch_size)
            n = len(models)
            a = (j % (n*n)) // n
            b = j % n
            playGame(Game(), models[a], models[b], modelTuples[a], modelTuples[b])
        #train each model on the model tuples
        #see this for many of the ideas used
        #http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-8.pdf
        print('training')
        for j in range(len(models)):
            #model that will be modified
            model = models[j]
            #static model that will be used for (most) evaluations
            targetModel = targetModels[j]
            tuples = modelTuples[j]

            for k in range(num_samples):
                samples = random.sample(tuples, sample_size)

                #get the value for a bunch of points
                #then update model in a batch
                for action, state, nextState, reward in samples:
                    #use target_model so we get a consistent evaluation of states
                    #use model for sort_model to reduce noise (double Q-learning)
                    val = discount * getStateValue(targetModel, nextState, sort_model=model) + reward
                    model.addDataLabel(state.getData() + action, val)
                model.batchedUpdate()

        #save models after each epoch
        for j in range(len(models)):
            filename = saveDir + '/' + 'model-' + name + str(j) + '.h5'
            models[j].saveModel(filename)

    #after training is done, let the user try out one of the models
    playWithUser(Game, models[0])

if __name__ == "__main__":
    train(TicTacToe, num_models=3, num_epochs=20)
