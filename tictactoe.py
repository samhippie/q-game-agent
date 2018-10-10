from game import Game

import sys

class TicTacToe(Game):
    input_shape = (2 + 36,)
    def __init__(self):
        self.board = [0] * 9
        self.turn = 1

    def getData(self):
        #flip the symbols if it's player 2's turn
        #so the current player always sees "my pieces" and "your pieces"
        #also uses one-hot encoding for each place
        pieceMap = {
            0: [1,0,0],
            1: [0,1,0] if self.turn == 1 else [0,0,1],
            2: [0,0,1] if self.turn == 1 else [0,1,0],
        }
        pieces = [d for ds in [pieceMap[i] for i in self.board] for d in ds]
        player = [1,0] if self.turn == 1 else [0,1]
        return player + pieces

    #returns 1 or 2 if 1 or 2 won, else None
    #also bumps up turn
    def takeTurn(self, action):
        place = 0
        for i in range(9):
            if action[i] == 1:
                self.board[i] = self.turn
                place = i
                break;
        self.turn = 1 if self.turn == 2 else 2
        return self.checkWinner(place)

    def checkWinner(self, place):
        #convenience function for checking 3 places
        def checkPlaces(a, b, c):
            if self.board[a] == self.board[b] and self.board[b] == self.board[c]:
                return self.board[a]
            return None

        #tie
        if 0 not in self.board:
            return -1

        #diagonal
        if place in [0, 4, 8]:
            winner = checkPlaces(0, 4, 8)
            if winner:
                return winner
        #antidiagonal
        if place in [2, 4, 6]:
            winner = checkPlaces(2, 4, 6)
            if winner:
                return winner
        #row
        row = place // 3
        winner = checkPlaces(3*row, 3*row + 1, 3*row + 2)
        if winner:
            return winner
        #column
        col = place % 3
        winner = checkPlaces(col, 3 + col, 6 + col)
        if winner:
            return winner
        #no one won, return None
        return None

    def printBoard(self, file=sys.stdout):
        for r in range(3):
            for c in range(3):
                print(self.board[r * 3 + c], end="", file=file)
            print(file=file)

    def getActions(self):
        moves = [i for i in range(9) if self.board[i] == 0]
        actions = []
        for move in moves:
            action = [0] * 9
            action[move] = 1
            actions.append(action)
        return actions

    #converts action to user-readable string
    def actionToString(self, action):
        move = 0
        for i in range(len(action)):
            if action[i] == 1:
                move = i
                break
        return str(move + 1)


