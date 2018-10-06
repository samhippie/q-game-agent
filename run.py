#!/usr/bin/env python3

import random

from tictactoe import TicTacToe
from checkers import Checkers
from agent import train

#plays the game with two completely random models
#for testing
def dummyAgent(Game, verbose=False):
    game = Game()
    result = None
    while result == None:
        if(verbose):
            print()
            game.printBoard()

        actions = game.getActions()
        result = game.takeTurn(random.choice(actions))

    if verbose:
        print()
        game.printBoard()
        print()
        print("result", result)

if __name__ == '__main__':
    train(Checkers, 'checkers-limit',
            num_models=1,
            epoch_size=1000, num_epochs=1000,
            sample_size=1000, num_samples=10,
            random_epochs=0,
            saveDir='/home/sam/scratch/tflow', loadModels=True)
    """
    train(TicTacToe, 'tictactoe-small',
            num_models=1,
            epoch_size=1000, num_epochs=100,
            sample_size=100, num_samples=10,
            saveDir='/home/sam/scratch/tflow', loadModels=True)
    """
