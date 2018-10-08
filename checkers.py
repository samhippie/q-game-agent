#!/usr/bin/env python3

import collections
import random

from game import Game

#TODO do something more elegant than a top-level constant
SIZE = 4

class Checkers(Game):
    #state: 5 bits per (valid) board square in state
    #only half of squares are valid

    #action: 64 bits for src, 64 bits for dst in action

    #counter: and we have 8 bits to count the number of turns
    #after 256 turns, the game ends in a tie
    input_shape = ((5 * SIZE * SIZE)/2 + 2 * SIZE * SIZE + 8,)

    def __init__(self, size=8):
        #8x8, [0] is 1a, dark
        #[63] is 8h, dark
        self.board = [0] * SIZE * SIZE
        #init the two sides
        #always leave gap of 2 in the middle
        for i in range(0, ((SIZE - 2) // 2) * SIZE):
            r = i // SIZE
            c = i % SIZE
            if r % 2 == i % 2:
                self.board[i] = 1
                self.board[SIZE*SIZE-1 - i] = 2

        self.turn = 1
        #simplest way to determine winner is to count the captures
        #we actually count down as it's simpler
        numPieces = ((SIZE - 2) * SIZE) // 4
        self.numCaptures = [numPieces,numPieces]
        #on capture, the dst is saved
        #if this is not None, and there are no jumps, then passing is
        #the only legal move
        self.justCaptured = None

        #<THIS IS NO LONGER USED>
        #track how often each state appears
        #if a previous state occurs 4 times in a game, the game is a tie
        self.prevStateCount = collections.defaultdict(int)
        #</THIS IS NO LONGER USED>

        #count the number of turns remaining
        #after so many turns, the game ends in a draw
        #this gives a markov-y way of preventing endless games
        self.turnCount = 255

    #returns a representation of the board state
    #oriented so each player sees themself as player 1
    def getData(self):
        #only return the playable squares
        squares = []
        #one hot representation
        pieceMap = {
            0 : numToOneHot(0, 5),
            1 : numToOneHot(1, 5) if self.turn == 1 else numToOneHot(2, 5),
            2 : numToOneHot(2, 5) if self.turn == 1 else numToOneHot(1, 5),
            3 : numToOneHot(3, 5) if self.turn == 1 else numToOneHot(4, 5),
            4 : numToOneHot(4, 5) if self.turn == 1 else numToOneHot(3, 5),
        }
        """
        #one-hot with a bit added for kinged pieces
        pieceMap = {
            0 : [1,0,0,0],
            1 : [0,1,0,0] if self.turn == 1 else [0,0,1,0],
            2 : [0,0,1,0] if self.turn == 2 else [0,1,0,0],
            3 : [0,1,0,1] if self.turn == 1 else [0,0,1,1],
            4 : [0,0,1,1] if self.turn == 2 else [0,1,0,1],
        }
        """
        #iterate over each piece in order, (reverse order for player 2)
        indices = range(SIZE*SIZE) if self.turn == 1 else range(SIZE*SIZE-1, -1, -1)
        for i in indices:
            r = i // SIZE
            c = i % SIZE
            if r % 2 == c % 2:
                squares = squares + pieceMap[self.board[i]]

        #the number of turns remaining
        #should this be one-hot?
        squares = squares + numToBinary(self.turnCount, 8)

        return squares

    #shows black as 1, red as 2, black king as !, red king as &
    #on a board with chess-style notations
    def printBoard(self):
        pieceMap = {
            0: '_',
            1: '1',
            2: '2',
            3: '!',
            4: '&',
        }
        #turn counter
        print('turns remaining:', self.turnCount)
        #top label
        cols = [chr(ord('a') + c) for c in range(SIZE)]
        print('  ', end='')
        for c in cols:
            print(c + ' ', end='')
        print()
        rows = '12345678'
        #backwards so that 0,0 (a1) is in the bottom right corner
        for r in range(SIZE-1, -1, -1):
            #left label
            print(rows[r] + '|', end='')
            #pieces
            for piece in self.board[r * SIZE : (r+1) * SIZE]:
                print(pieceMap[piece] + '|', end='')
            #right label
            print(rows[r])
        #bottom label
        print('  ', end='')
        for c in cols:
            print(c + ' ', end='')
        print()

    #applies the action to the board
    #adjusting self.turn as needed
    def takeTurn(self, action):
        #sometimes passing is a legal move, represented by all 0s
        if not 1 in action:
            self.turn = 1 if self.turn == 2 else 2
            self.justCaptured = None
            return None
        #forfeit is all 1s
        #first 2 are 1 -> all must be 1
        if action[1] == 1 and action[2] == 1:
            return 1 if self.turn == 2 else 2

        src = oneHotToNum(action[0 : SIZE*SIZE])
        dst = oneHotToNum(action[SIZE*SIZE : 2 * SIZE*SIZE])
        #player 2 submits flipped inputs
        if self.turn == 2:
            src = SIZE*SIZE-1 - src
            dst = SIZE*SIZE-1 - dst

        srcR = src // SIZE
        srcC = src % SIZE
        dstR = dst // SIZE
        dstC = dst % SIZE

        #if there is a capture, we should not advance the turn
        #and we should save what piece did the capturing
        captured = False
        #captured something
        if abs(dstR - srcR) == 2:
            capR = srcR + (dstR - srcR) // 2
            capC = srcC + (dstC - srcC) // 2
            self.board[capR * SIZE + capC] = 0
            captured = True

        #swap src and dst
        piece = self.board[src]
        self.board[src] = 0
        self.board[dst] = piece

        #king me
        if self.turn == 1 and dstR == SIZE-1:
            self.board[dst] = 3
        elif self.turn == 2 and dstR == 0:
            self.board[dst] = 4

        if captured:
            #so getActions knows what piece must move
            self.justCaptured = dst
            #check for win
            self.numCaptures[self.turn-1] -= 1
            if self.numCaptures[self.turn-1] == 0:
                return self.turn
        else:
            self.justCaptured = None
            #now it's the other player's turn
            self.turn = 1 if self.turn == 2 else 2

        #<THIS IS NO LONGER USED>
        #check for too many repeated states
        #if we repeat too many times, end the game with a tie
        """
        state = tuple(self.board)
        self.prevStateCount[state] += 1
        if self.prevStateCount[state] >= 4:
            return -1
        """
        #</THIS IS NO LONGER USED>

        #after so many turns, end the game in a tie
        self.turnCount -= 1
        if self.turnCount <= 0:
            return -1

        return None


    #return the list of all legal actions for the current player
    def getActions(self):
        #iterate over each piece
        #find out its legal moves
        #for each legal move, add representation to action
        #representation is just be start and end,
        #which would be 2 64-bit one-hot representations

        #use all 0s to pass
        #which is fine as what the network sees doesn't matter,
        #as passing will never be compared to another action's value


        #if jumpActions is non-empty, then all normActions are illegal
        normActions = []
        jumpActions = []
        #just captured => only one piece can move
        if self.justCaptured:
            pieceIndices = [self.justCaptured]
        else:
            pieceIndices = range(len(self.board))
        for i in pieceIndices:
            piece = self.board[i]
            #check side of piece
            ownPieces = [1,3] if self.turn == 1 else [2,4]
            otherPieces = [2,4] if self.turn == 1 else [1,3]
            if piece not in ownPieces:
                continue
            row = i // SIZE
            col = i % SIZE
            #king pieces, don't have to worry about orientation
            if piece in [3,4]:
                dirs = [(row+1,col+1), (row+1,col-1), (row-1,col+1), (row-1,col-1)]
            #other pieces only move one way
            elif piece == 1:
                dirs = [(row+1,col+1), (row+1,col-1)]
            elif piece == 2:
                dirs = [(row-1,col+1), (row-1,col-1)]
            src = i
            normDsts = []
            jumpDsts = []
            #4 directions
            for r,c in dirs:
                #keep it in the board
                if r < 0 or r >= SIZE or c < 0 or c >= SIZE:
                    continue
                index = r * SIZE + c
                #can't go to occupied space
                if self.board[index] in ownPieces:
                    continue
                if self.board[index] in otherPieces:
                    #check if we can jump over and capture
                    nextR = row + 2*(r - row)
                    nextC = col + 2*(c - col)
                    nextIndex = nextR * SIZE + nextC
                    if nextR >= 0 and nextR < SIZE and nextC >= 0 and nextC < SIZE and self.board[nextIndex] == 0:
                        jumpDsts.append(nextIndex)
                    else:
                        continue
                #otherwise, the move is fine
                elif len(jumpActions) == 0:# if there is a jump, don't bother looking at normal moves
                    normDsts.append(index)

            #if there is 1 jump, then there can only be jumps
            if len(jumpDsts) > 0:
                dsts = jumpDsts
                actions = jumpActions
            else:
                dsts = normDsts
                actions = normActions
            for d in dsts:
                if self.turn == 1:
                    actions.append(numToOneHot(src, SIZE*SIZE) +
                            numToOneHot(d, SIZE*SIZE))
                else:
                    actions.append(numToOneHot(SIZE*SIZE-1 - src, SIZE*SIZE) +
                            numToOneHot(SIZE*SIZE-1 - d, SIZE*SIZE))
        #normal turn
        if self.justCaptured == None:
            if len(jumpActions) == 0:
                actions = normActions
            else:
                actions = jumpActions
        else:
            if len(jumpActions) == 0:
                #pass, no legal moves
                actions = [[0] * 2 * SIZE * SIZE]
            else:
                actions = jumpActions
        #no legal moves means you forfeit
        #all 1s is a forfeit
        if len(actions) == 0:
            actions = [[1] * 2 * SIZE * SIZE]
        return actions

    #converts action to human-readable string
    #this uses chess notation, not standard checkers notation
    def actionToString(self, action):
        if not 1 in action:
            return "Pass (no legal moves)"
        if action[0] == 1 and action[1] == 1:
            return "Forfeit (no legal moves)"

        src = oneHotToNum(action[0 : SIZE*SIZE])
        dst = oneHotToNum(action[SIZE*SIZE : 2 * SIZE*SIZE])
        #if player is player 2, we need to un-flip the actions
        if self.turn == 2:
            src = SIZE*SIZE-1 - src
            dst = SIZE*SIZE-1 - dst

        srcR = src // SIZE
        srcC = src % SIZE
        dstR = dst // SIZE
        dstC = dst % SIZE
        #I'm using chess notation because it's my program and that how I want it
        #col 0 -> a
        #row 0 -> 1
        srcLabel = chr(ord('a') + srcC) + str(srcR + 1)
        dstLabel = chr(ord('a') + dstC) + str(dstR + 1)
        return srcLabel + ' -> ' + dstLabel

#converts a number to a one-hot representation
def numToOneHot(num, size):
    xs = [0] * size
    xs[num] = 1
    return xs

#converts a one-representation back to a number
def oneHotToNum(oneHot):
    for i in range(len(oneHot)):
        if oneHot[i] == 1:
            return i
    return 0

#converts a number to a binary representation, little endian
def numToBinary(num, size):
    xs = []
    for i in range(size):
        xs.append(num & 1)
        num >>= 1
    return xs


if __name__ == '__main__':
    c = Checkers()
    c.turn = 1
    c.printBoard()
    for a in c.getActions():
        print(c.actionToString(a))
