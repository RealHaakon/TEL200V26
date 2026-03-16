class Board:
    """
    A simple tic tac toe game featuring an AI.

    Features
    - Display
    - Make / Unmake moves
    - Movegenerator
    - Win detection

    Ai
    - Negamax
    - Alpha beta pruning
    - Bullying (for fun)
    """
    def __init__(self):
        # Bitboards
        self.occupiedSquares = 0b000000000
        self.friendlySquares = 0b000000000
        self.turn = True

        # Masks representing possible three in a rows
        self.winMasks = [
            0b111000000,
            0b000111000,
            0b000000111,
            0b100100100,
            0b010010010,
            0b001001001,
            0b100010001,
            0b001010100,
        ]

        # Tracking data during search
        self.bestMove = None
        self.count = 0

    def display(self):
        """
        Renders the current state of the board in the console
        """
        # fetch data
        opponentSquares = self.friendlySquares ^ self.occupiedSquares
        if (self.turn):
            friendly = "X"
            opposing = "O"
        else:
            friendly = "O"
            opposing = "X"
        # Generate a list from the board
        squares = []
        for i in range(0, 9):
            if (self.friendlySquares >> i) & 1 == 1:
                squares.append(friendly)
            elif (opponentSquares >> i) & 1 == 1:
                squares.append(opposing)
            else:
                squares.append(" ")

        # Render the board
        if (self.turn):
            print("X to move")
        else:
            print("O to move")
        for i in range(0, 3):
            sq1 = squares[i*3 + 0]
            sq2 = squares[i*3 + 1]
            sq3 = squares[i*3 + 2]
            print("-"*13)
            print(f"| {sq1} | {sq2} | {sq3} |")
        print("-"*13)

    def makeMove(self, posIndex):
        """
        Place a piece belonging to the current player
        in the square with a related index
        -------------
        | 0 | 1 | 2 |
        -------------
        | 3 | 4 | 5 |
        -------------
        | 6 | 7 | 8 |
        -------------
        """
        # add bit to masks
        self.friendlySquares ^= 1 << posIndex
        self.occupiedSquares ^= 1 << posIndex

        # flip turn
        self.turn = not self.turn

        # flip friendly bits
        self.friendlySquares ^= self.occupiedSquares

    def unmakeMove(self, posIndex):
        """
        Restores a postion before the previous move was played.
        Removes the piece belonging to the current player
        in the square with a related index.
        -------------
        | 0 | 1 | 2 |
        -------------
        | 3 | 4 | 5 |
        -------------
        | 6 | 7 | 8 |
        -------------
        """
        # flip friendly bits
        self.friendlySquares ^= self.occupiedSquares

        # remove bit from masks
        self.friendlySquares ^= 1 << posIndex
        self.occupiedSquares ^= 1 << posIndex

        # flip turn
        self.turn = not self.turn

    def generateMoves(self):
        """
        Returns a list of the indexes of the empty squares on the board
        """
        legalMoves = []
        for i in range(0, 9):
            # Square is not occupied
            if ((self.occupiedSquares >> i) & 1 == 0):
                legalMoves.append(i)

        return legalMoves

    def AI(self):
        """
        A simple wrapper for the Negamax AI. Negamax gets called
        with a lot of parameters, and this makes it easier to use.
        """

        # Reset variables
        self.bestMove = None
        self.count = 0

        # search the position
        score = self.negaMax(0, -100_000, 100_000)
        print("nodes: ", self.count)

        # Some bullying is fun (triggers for guaranteed victory)
        if (0 < score):
            print("You are fucked!")

        return self.bestMove

    def negaMax(self, ply, alpha, beta):
        """
        Recursively calls Negamax to search all nodes in the game tree.
        Speed up using alpha beta pruning, reducing the total node count
        from startpos from 50000 -> 20000
        """
        # Node count for tracking performance
        self.count += 1

        # Win detection
        if (self.hasLost()):
            return - (1000 - ply)

        # Generate moves
        moves = self.generateMoves()
        if len(moves) == 0:
            # Terminal leaf node
            return 0

        # bestScore
        bestScore = -10000

        for move in moves:
            # iterative negamax
            self.makeMove(move)
            score = - self.negaMax(ply+1, -beta, -alpha)
            self.unmakeMove(move)

            # track lowebound score
            alpha = max(alpha, score)

            # fail high node
            if beta <= alpha:
                return alpha

            # Update best score
            if bestScore < score:
                bestScore = score
                # In rootpos keep track of the best move
                if (ply == 0):
                    self.bestMove = move

        return bestScore

    def hasLost(self):
        """
        Checks if the opposing player has won. Because the make move method
        flips the turn after making a move, it is impossible for the current
        player to have won.
        """
        # Check if the opponent has three in a row
        opponentSquares = self.friendlySquares ^ self.occupiedSquares
        for winMask in self.winMasks:
            if (opponentSquares & winMask == winMask):
                return True

        return False


def PlayGame(start=True):
    """
    A simple game loop to play a game against the algorithm.
    """
    # Create enviroment
    board = Board()
    yourTurn = start

    while True:
        board.display()

        # No moves means there is a draw
        legalMoves = board.generateMoves()
        if (len(legalMoves) == 0):
            print("Tie")
            break

        # Make your move
        if (yourTurn):
            move = -1
            while move not in legalMoves:
                move = int(input("Move: "))
        # AI makes a move
        else:
            board.AI()
            move = board.bestMove

        # Make the move on the board
        board.makeMove(move)

        # It is now the other players turn so you can check if
        # you won by checking if they lost
        # (AI is unbeatable so i didnt bother printing wins)
        if (board.hasLost()):
            board.display()
            print("You lost")
            break

        yourTurn = not yourTurn


if __name__ == "__main__":
    playerStarting = False
    while True:
        msg = input("New game Y/N: ")
        if msg == "Y":
            PlayGame(playerStarting)
        if msg == "N":
            break

        playerStarting = not playerStarting
