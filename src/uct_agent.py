# TODO - optimize for performance
from __future__ import annotations

import math
from random import choice

import chess

from .agent import Agent


class UCTNode:
    def __init__(self, game_state: str, parent=None):
        self.game_state = game_state  # FEN string
        self.is_expanded = False  # False indicates that this is a leaf node
        self.parent = parent  # Optional[UCTNode]
        self.children = {}  # Dict[move, UCTNode]
        self.total_value = 0  # int (sum of total rewards)
        self.number_visits = 0  # int

    def select_leaf(self) -> UCTNode:
        current = self
        # Traverse the tree, selecting the best child, until you reach a leaf
        while current.is_expanded:
            current = current.best_child()
        return current

    def expand(self):
        legal_moves = self._get_legal_moves(self.game_state)
        if len(legal_moves) == 0:
            return # Do not expand node that is a terminal state
        # Mark leaf node as expanded
        self.is_expanded = True
        # EAGERLY add a child node for each legal move
        for move in legal_moves:
            self.add_child(move)

    def backup(self, value_estimate):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value_estimate
            current = current.parent

    def add_child(self, move: str):
        child_state = self._get_next_state(self.game_state, move)
        self.children[move] = UCTNode(child_state, parent=self)

    def best_child(self) -> UCTNode:
        return max(self.children.values(), key=lambda node: node.Q() + node.U())

    def U(self) -> float:
        # TODO - add exploration parameter c
        # Add 1 to avoid numerical errors
        return math.sqrt(math.log(self.parent.number_visits + 1) / (self.number_visits + 1))

    def Q(self) -> float:
        return self.total_value / (1 + self.number_visits)

    @staticmethod
    def _get_legal_moves(fen: str) -> list:
        legal_moves = list(chess.Board(fen).legal_moves)
        return [str(move) for move in legal_moves]

    @staticmethod
    def _get_next_state(fen: str, move: str) -> str:
        board = chess.Board(fen)
        board.push_uci(move)
        return board.fen()


class UCTAgent(Agent):
    """Upper Confidence Bounds for Trees (UCT) algorithm applied to chess.

    The UCT algorithm is a specific implementation of the more general Monte
    Carlo Tree Search (MCTS) algorithm. UCT uses the Upper-Confidence-Bound
    (UCB) action-selection method for the tree policy. For the rollout policy
    we randomly select actions from a list of legal moves.
    """

    def __init__(self, name: str, is_white=True, iterations: int = 10):
        super().__init__(name, is_white)
        self.iterations = iterations

    def observe(self, reward: int, observation: str) -> str:
        # Observation is a string in Forsyth-Edwards Notation (FEN)
        # Returns an action as a string in algebraic notation (e.g. e3f2)
        best_action = self._UCT_search(observation, num_iterations=self.iterations)
        return best_action

    def _UCT_search(self, game_state: str, num_iterations: int) -> str:
        # Game state is a string in Forsyth-Edwards Notation (FEN)
        # Returns an action as a string in algebraic notation (e.g. e3f2)
        root = UCTNode(game_state)
        for _ in range(num_iterations):
            leaf = root.select_leaf()
            leaf.expand()
            simulation_result = self._rollout(leaf.game_state)  # 1=win, -1=loss, 0=draw
            leaf.backup(simulation_result)
        # return the action that was taken the most times
        return max(root.children.items(), key=lambda item: item[1].number_visits)[0]

    def _rollout(self, state: str) -> int:
        # Random rollout
        # Take as input a state (FEN string)
        # return an integer. 1 = win. -1 = loss. 0 = draw

        def _get_random_move(board: chess.Board) -> str:
            legal_moves = list(board.legal_moves)
            legal_moves_str = [str(move) for move in legal_moves]
            return choice(legal_moves_str)

        def _get_reward(board) -> int:
            # Get the result of the game. E.g. ('1-0', '0-1, or '1/2-1/2')
            result = board.result()
            # Determine the reward from the simulated rollout.
            if result == '1-0':
                reward = 1
            elif result == '0-1':
                reward = -1
            elif result == '1/2-1/2':
                reward = 0
            else:
                raise Exception('Unknown end game state')
            # If Agent is black, then flip reward
            if not self.is_white:
                reward *= -1
            return reward

        board = chess.Board(state)
        is_game_over = board.is_game_over()
        while not is_game_over:
            move = _get_random_move(board)
            board.push_uci(move)
            is_game_over = board.is_game_over()
        return _get_reward(board)
