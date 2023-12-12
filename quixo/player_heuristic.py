from game import Game, Move, Player
import random
import numpy as np
from copy import deepcopy
from utils1 import (
    move1,
    possible_takes,
)


class MyPlayer(Player):
    def __init__(self, player_index) -> None:
        super().__init__()
        self.index = player_index

    def heuristic(self, board, player):
        """Evaluate the board for the current player"""
        if player > 1:
            raise ValueError("Player must be 0 or 1")

        score_current = 0

        # weights
        corner_weight = 10
        edges_center_weight = 3
        core_weight = 5
        almost_complete_lines_weight = 100

        # owned cornes
        score_current += corner_weight if board[0][0] == player else 0
        score_current += corner_weight if board[0][4] == player else 0
        score_current += corner_weight if board[4][0] == player else 0
        score_current += corner_weight if board[4][4] == player else 0

        # edges cetner
        score_current += edges_center_weight if board[0][2] == player else 0
        score_current += edges_center_weight if board[2][0] == player else 0
        score_current += edges_center_weight if board[2][4] == player else 0
        score_current += edges_center_weight if board[4][2] == player else 0

        # core
        for i in range(1, 4):
            for j in range(1, 4):
                score_current += core_weight if board[i][j] == player else 0

        # almost complete lines
        for i in range(5):
            if np.count_nonzero(board[i] == player) == 4:
                score_current += almost_complete_lines_weight
            if np.count_nonzero(board[:, i] == player) == 4:
                score_current += almost_complete_lines_weight

        return score_current

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])

        current_board = game._board
        current_heuristic = self.heuristic(current_board, self.index)

        b = deepcopy(current_board)
        b_heuristic = current_heuristic

        # find best move depth 1
        for from_position, slide in possible_takes(current_board, self.index):
            b = deepcopy(current_board)
            if move1(b, from_position, slide, self.index):
                heuristic = self.heuristic(b, self.index)
                if heuristic > b_heuristic:
                    b_heuristic = heuristic
                    from_pos = from_position
                    move = slide

        return from_pos, move
