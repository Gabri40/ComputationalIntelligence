from game import Game, Move, Player
import random
import numpy as np
from utils import *
from tqdm import tqdm


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class MinimaxPlayer(Player):
    def __init__(self, depth: int, player_index) -> None:
        super().__init__()
        self.depth = depth
        self.player_index = player_index

    def mid_game_heuristic(self, board):
        boardeval = 0

        # max number in a row col or diag for each player
        mp, mo = 0, 0
        for i in range(5):
            mp = max(mp, np.count_nonzero(board[i] == self.player_index))
            mo = max(mo, np.count_nonzero(board[i] == 1 - self.player_index))
            mp = max(mp, np.count_nonzero(board[:, i] == self.player_index))
            mo = max(mo, np.count_nonzero(board[:, i] == 1 - self.player_index))
        mp = max(mp, np.count_nonzero(np.diag(board) == self.player_index))
        mo = max(mo, np.count_nonzero(np.diag(board) == 1 - self.player_index))

        boardeval += 5**mp - 5**mo

        # piece count
        cp = np.count_nonzero(board.flatten() == self.player_index)
        op = np.count_nonzero(board.flatten() == 1 - self.player_index)
        boardeval += 2**cp - 2**op

        # # Core count
        # for i in range(1, 4):
        #     for j in range(1, 4):
        #         boardeval += (
        #             1
        #             if board[i][j] == self.player_index
        #             else -1
        #             if board[i][j] == 1 - self.player_index
        #             else 0
        #         )

        # Edge count
        edge_positions = (
            [(0, i) for i in range(1, 4)]
            + [(4, i) for i in range(1, 4)]
            + [(i, 0) for i in range(1, 4)]
            + [(i, 4) for i in range(1, 4)]
        )
        for x, y in edge_positions:
            boardeval += (
                2
                if board[x][y] == self.player_index
                else -2
                if board[x][y] == 1 - self.player_index
                else 0
            )

        # # Corner count
        # corner_positions = [(0, 0), (0, 4), (4, 0), (4, 4)]
        # for x, y in corner_positions:
        #     boardeval += (
        #         1
        #         if board[x][y] == self.player_index
        #         else -1
        #         if board[x][y] == 1 - self.player_index
        #         else 0
        #     )

        return boardeval

    def alphabeta(self, board, depth, alpha, beta, maximizing):
        boardeval = evaluate_winner(board)
        if boardeval == self.player_index:
            return 700 + depth**2
        elif boardeval == 1 - self.player_index:
            return -700 - depth**2
        else:
            boardeval = self.mid_game_heuristic(board)

        if depth == 0:
            return boardeval

        if maximizing:
            maxEval = float("-inf")
            for action in get_all_possible_actions(board, self.player_index):
                next_state, _ = try_move(action[0], action[1], board, self.player_index)
                eval = self.alphabeta(next_state, depth - 1, alpha, beta, False)
                maxEval = max(maxEval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return maxEval

        else:
            minEval = float("inf")
            for action in get_all_possible_actions(board, 1 - self.player_index):
                next_state, _ = try_move(
                    action[0], action[1], board, 1 - self.player_index
                )
                eval = self.alphabeta(next_state, depth - 1, alpha, beta, True)
                minEval = min(minEval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return minEval

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        board = game.get_board()
        bestEval = float("-inf")
        bestAction = None
        for action in get_all_possible_actions(board, self.player_index):
            next_state, _ = try_move(action[0], action[1], board, self.player_index)
            eval = self.alphabeta(
                next_state, self.depth, float("-inf"), float("inf"), False
            )
            if eval > bestEval:
                bestEval = eval
                bestAction = action

        return (bestAction[0][1], bestAction[0][0]), bestAction[1]


if __name__ == "__main__":
    wins = 0
    games = 50
    for i in tqdm(range(games)):
        game = Game()
        player1 = MinimaxPlayer(5, 0)
        player2 = RandomPlayer()
        if game.play(player1, player2) == 0:
            wins += 1
    print(wins / games)


# seems to be good enought at around 55% winrate it doesnt improve much after that
