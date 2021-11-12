from random import choice

import chess
import chess.svg


class ActionSpace:
    def __init__(self, board):
        self.board = board

    def sample(self) -> str:
        """
        Returns a random legal move as a string in algebraic notation
        :return: move as a string
        """
        return choice(self.available_actions())

    def available_actions(self) -> list[str]:
        """
        Returns a list of legal moves as a list of strings where the
        moves are in algebraic notation
        :return: list of moves
        """
        legal_moves = list(self.board.legal_moves)
        legal_moves_str = [str(move) for move in legal_moves]
        return legal_moves_str


class ChessEnvV1:
    """
    Chess environment. Player is always white. Opponent plays randomly.
    """

    def __init__(self):
        self.board = chess.Board()
        self.action_space = ActionSpace(self.board)

    def _observation(self) -> str:
        """
        Returns the state of the game in Forsyth-Edwards Notation (FEN)
        :return: observation
        """
        return self.board.fen()

    def _is_done(self) -> bool:
        """
        :return: Returns True if the game is over, else False
        """
        return self.board.is_game_over()

    def _reward(self) -> int:
        """
        Returns a reward from white's perspective.
        Game not over = 0
        Win = +1
        Loss = -1
        Draw = 0
        :return: reward
        """
        if not self._is_done():
            reward = 0
        else:
            result = self.board.result()
            if result == '1-0':
                reward = 1
            elif result == '0-1':
                reward = -1
            elif result == '1/2-1/2':
                reward = 0
            else:
                raise Exception('Unknown end game state')
        return reward

    def reset(self) -> str:
        """
        Resets the game and returns an observation
        :return: observation
        """
        return self._observation()

    def render_image(self, **kwargs):
        return chess.svg.board(self.board, **kwargs)

    def render_text(self):
        print('\n')
        print(self.board)
        print('\n')

    def step(self, action: str) -> tuple[str, int, bool]:
        """
        :param action: action in algebraic notation
        :return: observation, reward, is_done
        """
        self.board.push_uci(action)
        is_done = self._is_done()
        if is_done:
            obs = self._observation()
            reward = self._reward()
        else:
            # let opponent make a random move
            black_move = self.action_space.sample()
            self.board.push_uci(black_move)
            obs = self._observation()
            is_done = self._is_done()
            reward = self._reward()

        return obs, reward, is_done
