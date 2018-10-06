class Game:
    input_shape = (0,)

    def getData(self):
        raise Error('get data not implemented')

    #returns 1 or 2 if 1 or 2 won, else None
    def takeTurn(self, action):
        raise Error('take turn not implemented')

    def printBoard(self):
        raise Error('print board not implemented')

    #gets the legal actions for the current turn
    def getActions(self):
        raise Error('get actions not implemented')

    #converts action to human-readable string
    def actionToString(action):
        raise Error('action to string not implemented')

