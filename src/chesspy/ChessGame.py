import chess
import numpy as np

from Game import Game

# noinspection SpellCheckingInspection

# Code motivated/based on https://github.com/Zeta36/chess-alpha-zero/blob/master/src/chess_zero/env/chess_env.py

# input planes
pieces_order = 'KQRBNPkqrbnp'  # 12x8x8
castling_order = 'KQkq'  # 4x8x8
# fifty-move-rule             # 1x8x8
# en en_passant               # 1x8x8
ind = {pieces_order[i]: i for i in range(12)}


class ChessGame(Game):
    """
    This class specifies the Chess Game class. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    """

    def __init__(self):
        # TODO move to a location to be static
        self.all_possible_moves = create_uci_labels()
        # self.winner = None  # type: Winner
        # self.resigned = False
        # self.result = None

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        # create input layers with fresh board
        # return create_input_planes(self.board.fen())
        return chess.Board()

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (18, 8, 8)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return len(self.all_possible_moves)

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        # TODO remove assert not required part for speed
        assert libPlayerToChessPlayer(board.turn) == player
        move = self.all_possible_moves[action]
        if not board.turn:  # black move from CanonicalForm
            move = str(mirror_action(chess.Move.from_uci(move)))

        if move not in getAllowedMovesFromBoard(board): # must be a pawn promotion
            # print(board)
            # print(move, " is not valid - ",self.all_possible_moves[action])
            move=move+self.all_possible_moves[action][-1:]
            # print("moveupdated:",move)
        board = board.copy()
        board.push(chess.Move.from_uci(move))
        return (board, -player)

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        # TODO remove assert not required part for speed
        assert libPlayerToChessPlayer(board.turn) == player
        current_allowed_moves = np.array(getAllowedMovesFromBoard(board))
        # TODO find a better way
        validMoves = np.isin(np.array(self.all_possible_moves), current_allowed_moves).astype(int)

        assert np.sum(validMoves) == len(current_allowed_moves)
        return validMoves

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        r = board.result()
        if r == "1-0":
            return player
        elif r == "0-1":
            return -player
        elif r == "1/2-1/2":
            return 0.001  # TODO how small is better?
        else:
            return 0

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        # TODO remove assert not required part for speed
        assert libPlayerToChessPlayer(board.turn) == player
        if player == 1:
            return board
        else:
            return board.mirror()

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board, pi)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        # TODO maybe player move matters?
        fen = board.fen()
        # l = fen.rindex(' ', fen.rindex(' '))
        # return fen[0:l]
        parts = board.fen().split(' ')
        return parts[0] + ' ' + parts[2] + ' ' + parts[3]

    @staticmethod
    def display(board):
        print(board)

    def toArray(self, board):
        fen = board.fen()
        fen = maybe_flip_fen(fen, is_black_turn(fen))
        return all_input_planes(fen)


def mirror_action(action):
    return chess.Move(chess.square_mirror(action.from_square), chess.square_mirror(action.to_square))


def getAllowedMovesFromBoard(board):
    return [move.uci() for move in board.legal_moves]


# reverse the fen representation as if black is white and vice-versa
def maybe_flip_fen(fen, flip=False):
    if not flip:
        return fen
    foo = fen.split(' ')
    rows = foo[0].split('/')

    return "/".join([swapall(row) for row in reversed(rows)]) \
           + " " + ('w' if foo[1] == 'b' else 'b') \
           + " " + "".join(sorted(swapall(foo[2]))) \
           + " " + foo[3] + " " + foo[4] + " " + foo[5]


def is_black_turn(fen):
    return fen.split(" ")[1] == 'b'


def all_input_planes(fen):
    current_aux_planes = aux_planes(fen)

    history_both = to_planes(fen)

    ret = np.vstack((history_both, current_aux_planes))
    assert ret.shape == (18, 8, 8)
    return ret


# Create layers for
# castling_order = 'KQkq'     # 4x8x8
# fifty-move-rule             # 1x8x8
# en en_passant               # 1x8x8

def aux_planes(fen):
    foo = fen.split(' ')

    en_passant = np.zeros((8, 8), dtype=np.float32)
    if foo[3] != '-':
        eps = alg_to_coord(foo[3])
        en_passant[eps[0]][eps[1]] = 1

    fifty_move_count = int(foo[4])
    fifty_move = np.full((8, 8), fifty_move_count, dtype=np.float32)

    castling = foo[2]
    auxiliary_planes = [np.full((8, 8), int('K' in castling), dtype=np.float32),
                        np.full((8, 8), int('Q' in castling), dtype=np.float32),
                        np.full((8, 8), int('k' in castling), dtype=np.float32),
                        np.full((8, 8), int('q' in castling), dtype=np.float32),
                        fifty_move,
                        en_passant]

    ret = np.asarray(auxiliary_planes, dtype=np.float32)
    assert ret.shape == (6, 8, 8)
    return ret


# create layers for pieces in order = 'KQRBNPkqrbnp'  # 12x8x8
def to_planes(fen):
    board_state = replace_tags_board(fen)
    pieces_both = np.zeros(shape=(12, 8, 8), dtype=np.float32)
    for rank in range(8):
        for file in range(8):
            v = board_state[rank * 8 + file]
            if v.isalpha():
                pieces_both[ind[v]][rank][file] = 1
    assert pieces_both.shape == (12, 8, 8)
    return pieces_both


def replace_tags_board(board_san):
    board_san = board_san.split(" ")[0]
    board_san = board_san.replace("2", "11")
    board_san = board_san.replace("3", "111")
    board_san = board_san.replace("4", "1111")
    board_san = board_san.replace("5", "11111")
    board_san = board_san.replace("6", "111111")
    board_san = board_san.replace("7", "1111111")
    board_san = board_san.replace("8", "11111111")
    return board_san.replace("/", "")


def alg_to_coord(alg):
    rank = 8 - int(alg[1])  # 0-7
    file = ord(alg[0]) - ord('a')  # 0-7
    return rank, file


def swapcase(a):
    if a.isalpha():
        return a.lower() if a.isupper() else a.upper()
    return a


def swapall(aa):
    return "".join([swapcase(a) for a in aa])


def create_uci_labels():
    """
    Creates the labels for the universal chess interface into an array and returns them
    :return:
    """
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for l1 in range(8):
        for n1 in range(8):
            destinations = [(t, n1) for t in range(8)] + \
                           [(l1, t) for t in range(8)] + \
                           [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                           [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)
    for l1 in range(8):
        l = letters[l1]
        for p in promoted_to:
            labels_array.append(l + '2' + l + '1' + p)
            labels_array.append(l + '7' + l + '8' + p)
            if l1 > 0:
                l_l = letters[l1 - 1]
                labels_array.append(l + '2' + l_l + '1' + p)
                labels_array.append(l + '7' + l_l + '8' + p)
            if l1 < 7:
                l_r = letters[l1 + 1]
                labels_array.append(l + '2' + l_r + '1' + p)
                labels_array.append(l + '7' + l_r + '8' + p)
    return labels_array


def libPlayerToChessPlayer(turn):
    return 1 if turn else -1
