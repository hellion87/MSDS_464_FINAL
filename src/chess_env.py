from random import choice
from datetime import date

import chess
import chess.svg
import chess.pgn
from .agent import Agent


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

    def __init__(self, opponent: Agent):
        self.opponent = opponent
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.game.headers["White"] = 'Agent'
        self.game.headers["Black"] = self.opponent.name
        self.game.headers["Date"] = str(date.today())
        self.node = self.game
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
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.game.headers["White"] = 'Agent'
        self.game.headers["Black"] = self.opponent.name
        self.game.headers["Date"] = str(date.today())
        self.node = self.game
        return self._observation()

    def render_image(self, **kwargs):
        return chess.svg.board(self.board, **kwargs)

    def render_text(self):
        print('\n')
        print(self.board)
        print('\n')

    def step(self, action: str) -> tuple[str, int, bool, chess.pgn.Game]:
        """
        :param action: action in algebraic notation
        :return: observation, reward, is_done
        """
        self.board.push_uci(action)
        self.node = self.node.add_variation(chess.Move.from_uci(action))  # Add game node to log
        is_done = self._is_done()
        obs = self._observation()
        if not is_done:
            # let opponent make a move
            black_move = self.opponent.observe(0, obs)
            self.board.push_uci(black_move)
            self.node = self.node.add_variation(chess.Move.from_uci(black_move))  # Add game node to log
            obs = self._observation()
            is_done = self._is_done()

        self.game.headers["Result"] = self.board.result()
        reward = self._reward()
        info = self.game
        return obs, reward, is_done, info
