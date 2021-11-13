from abc import ABC, abstractmethod
from random import choice

import chess
import chess.engine


class Agent(ABC):
    def __init__(self, name: str, is_white=True):
        self.name = name

    @abstractmethod
    def observe(self, reward: int, observation: str) -> str:
        """
        Abstract base class for implementing agents

        :param reward: Reward received at time step t. This is the reward that came from Action_{t-1}
        :param observation: FEN observation from time step t.
        :return: An action in algebraic notation.
        """
        pass


class RandomAgent(Agent):

    def __init__(self, name: str):
        super().__init__(name)

    def observe(self, reward: int, observation: str) -> str:
        """
        Given an observation, the agent returns a random action.
        """
        board = chess.Board(observation)
        legal_moves = list(board.legal_moves)
        legal_moves_str = [str(move) for move in legal_moves]
        return choice(legal_moves_str)


class StockfishAgent(Agent):

    def __init__(self, name: str, stockfish_path: str, elo: int = 2850):
        """
        :param name:
        :param stockfish_path: This is the path to your local stockfish binary.
        It might look something like this: '/opt/homebrew/bin/stockfish'
        :param elo: The desired ELO score for this bot. Valid range includes 1350 to 2850
        https://github.com/official-stockfish/Stockfish/blob/2046d5da30b2cd505b69bddb40062b0d37b43bc7/src/ucioption.cpp
        """
        assert(1350 <= elo <= 2850)
        super().__init__(name)
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({"UCI_LimitStrength": 1800})
        self.engine.configure({"UCI_Elo": elo})

    def observe(self, reward: int, observation: str) -> str:
        board = chess.Board(observation)
        result = self.engine.play(board, chess.engine.Limit(time=0.1))
        move = str(result.move)
        return move
