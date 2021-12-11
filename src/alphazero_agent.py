# TODO - optimize for performance
from __future__ import annotations

import math
from random import choice
from typing import Dict

import chess
import numpy as np
#from .agent import Agent
from agent import Agent

# input planes
pieces_order = 'KQRBNPkqrbnp'  # 12x8x8
castling_order = 'KQkq'  # 4x8x8
# fifty-move-rule             # 1x8x8
# en en_passant               # 1x8x8
ind = {pieces_order[i]: i for i in range(12)}


class AZNode:
    def __init__(self, game_state: str, parent: AZNode = None, c=1.0):
        self.game_state = game_state  # FEN string
        self.is_expanded = False  # False indicates that this is a leaf node
        self.parent = parent
        self.children: Dict[str, AZNode] = {}  # Dict[move, AZNode]
        self.total_value: int = 0  # sum of total rewards
        self.number_visits: int = 0
        self.c: float = c  # exploration hyper-parameter for AZ. Increase c to explore.

    def select_leaf(self) -> AZNode:
        current = self
        # Traverse the tree, selecting the best child, until you reach a leaf
        while current.is_expanded:
            current = current.best_child()
        return current

    def expand(self):
        legal_moves = self._get_legal_moves(self.game_state)
        if len(legal_moves) == 0:
            return  # Do not expand node that is a terminal state
        # Mark leaf node as expanded
        self.is_expanded = True
        # EAGERLY add a child node for each legal move
        for move in legal_moves:
            self.add_child(move)

    def backup(self, value_estimate: int):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value_estimate
            current = current.parent
        current.number_visits += 1  # increment visit count for root

    def add_child(self, move: str):
        child_state = self._get_next_state(self.game_state, move)
        self.children[move] = AZNode(child_state, parent=self, c=self.c)

    def best_child(self) -> AZNode:
        best_node = None
        best_value = -np.inf
        for key in self.children:
            node = self.children[key]
            if node.total_value > best_value
                best_value = node.total_value
                best_node = node
        return best_node

    @staticmethod
    def _get_legal_moves(fen: str) -> list:
        legal_moves = list(chess.Board(fen).legal_moves)
        return [str(move) for move in legal_moves]

    @staticmethod
    def _get_next_state(fen: str, move: str) -> str:
        board = chess.Board(fen)
        board.push_uci(move)
        return board.fen()


class AZAgent(Agent):
    """Upper Confidence Bounds for Trees (AZ) algorithm applied to chess.

    The AZ algorithm is a specific implementation of the more general Monte
    Carlo Tree Search (MCTS) algorithm. AZ uses the Upper-Confidence-Bound
    (UCB) action-selection method for the tree policy. For the rollout policy
    we randomly select actions from a list of legal moves.
    """

    def __init__(self, name: str, nnet, game, is_white=True, iterations: int = 10, c=1):
        super().__init__(name, is_white)
        self.iterations = iterations
        self.c: float = c
        self.last_iter_root = None # root node from last iteration's search
        self.nnet = nnet
        self.game = game

    def observe(self, reward: int, observation: str) -> str:
        # Observation is a string in Forsyth-Edwards Notation (FEN)
        # Returns an action as a string in algebraic notation (e.g. e3f2)
        best_action = self._AZ_search(observation, num_iterations=self.iterations)
        return best_action

    def _AZ_search(self, game_state: str, num_iterations: int, return_probs=False):
        # Game state is a string in Forsyth-Edwards Notation (FEN)
        # Returns an action as a string in algebraic notation (e.g. e3f2)

        # Re-use the search tree from the last iteration if possible
        if self.last_iter_root and self._bfs(self.last_iter_root, game_state):
            root = self._bfs(self.last_iter_root, game_state)
        # Otherwise create a new tree from scratch
        else:
            root = AZNode(game_state, c=self.c)

        for _ in range(num_iterations):
            leaf = root.select_leaf()
            leaf.expand()
            simulation_result = self._rollout(leaf.game_state)  # 1=win, -1=loss, 0=draw
            leaf.backup(simulation_result)
        actions = [action for action in root.children]
        counts = np.array([root.children[key].number_visits for key in root.children])
        if not return_probs:
            # return the action that was taken the most times
            move = actions[np.argmax(counts)]
        else:
            sum_counts = np.sum(counts)
            probs = []
            for action in self.game.all_possible_moves: #to do: define me
                #if action was taken in state, append counts / sum_counts, else append 0
                if action in actions:
                    #find index in counts where corresponding to action
                    probs.append(counts[np.where(actions = action)] / sum_counts)
                else:
                    probs.append(0)
            move = np.array(probs)
    
        # Save the tree for the next search
        self.last_iter_root = root

        return move

    def _bfs(self, root: AZNode, fen: str):
        # Starting from root, search a tree for a node where the game state
        # matches the passed in FEN
        queue = [root]
        while len(queue) > 0:
            vertex = queue.pop(0)
            if vertex.game_state == fen:
                return vertex

            child_nodes = vertex.children.values()
            queue.extend(child_nodes)

    def _rollout(self, state: str) -> int:
        # Take as input a state (FEN string)
        # return value returned from self.nnet


        def _get_reward(board: chess.Board) -> int:
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
        if board.is_game_over():
            return _get_reward(board)
        else:
            encoded_state = self.encode_state(state)
            _, value = self.nnet(encoded_state)
            return -1 * value
   
    def all_input_planes(self, state):
        current_aux_planes = self.aux_planes(state)
        history_both = self.to_planes(state)
        ret = np.vstack((history_both, current_aux_planes))
        assert ret.shape == (18, 8, 8)
        return ret

    def aux_planes(self, state):
        foo = state.split(' ')
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
                            fifty_move, en_passant]

        ret = np.asarray(auxiliary_planes, dtype=np.float32)
        return ret

    def to_planes(self, state):
        board_state = self.replace_tags_board(state)
        pieces_both = np.zeros(shape=(12, 8, 8), dtype=np.float32)
        for rank in range(8):
            for file in range(8):
                v = board_state[rank * 8 + file]
                if v.isalpha():
                    pieces_both[ind[v]][rank][file] = 1
        return pieces_both

    def replace_tags_board(self, board_san):
        board_san = board_san.split(" ")[0]
        board_san = board_san.replace("2", "11")
        board_san = board_san.replace("3", "111")
        board_san = board_san.replace("4", "1111")
        board_san = board_san.replace("5", "11111")
        board_san = board_san.replace("6", "111111")
        board_san = board_san.replace("7", "1111111")
        board_san = board_san.replace("8", "11111111")
        return board_san.replace("/", "")

    def swapall(self, aa):
        return "".join([self.swapcase(a) for a in aa])

    def swapcase(self, a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a



    def encode_state(self, state):
        #if it's black's turn, reverse the board
        if state.split(" ")[1] == 'b':
            foo = state.split(' ')
            rows = foo[0].split('/')
            state = "/".join([self.swapall(row) for row in reversed(rows)]) \
                       + " " + ('w' if foo[1] == 'b' else 'b') \
                       + " " + "".join(sorted(self.swapall(foo[2]))) \
                       + " " + foo[3] + " " + foo[4] + " " + foo[5]
        return self.all_input_planes(state)

