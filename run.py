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
    train(Checkers, 'checkers-4',
            model_width=64,
            #alpha_steps={0: 0.1, 5: 0.01, 10: 0.001, 20: 0.0001},
            alpha_steps={0: 0.001},
            #discount_steps={0:0.7, 30: 0.9, 60: 0.99},
            discount_steps={0:0.99},
            epsilon_steps={0: 1, 5: 0.7, 20: 0.5, 50: 0.3},
            #epsilon_steps={0: 1},
            num_models=2,
            epoch_size=1000, num_epochs=1000,
            sample_size=1000, num_samples=100,
            play_at_end=False,
            saveDir='/home/sam/scratch/tflow', loadModels=False)
    """
    train(TicTacToe, 'tictactoe-small',
            model_width=64,
            alpha_steps={0: 0.01, 10: 0.001, 30: 0.0001},
            #alpha_steps={0: 0.001},
            discount_steps={0:0.5, 10: 0.7, 20: 0.9, 30: 0.99},
            #discount_steps={0:0.99},
            epsilon_steps={0: 1, 5: 0.7, 10: 0.5, 30: 0.3},
            #epsilon_steps={0: 0.3},
            num_models=1,
            epoch_size=1000, num_epochs=0,
            sample_size=100, num_samples=100,
            play_at_end=True,
            saveDir='/home/sam/scratch/tflow', loadModels=True)
    """
