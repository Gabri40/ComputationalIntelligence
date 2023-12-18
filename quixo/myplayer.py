import random
from game import Game, Move, Player
from random_player import RandomPlayer
from tqdm import tqdm
import pickle
from utils import get_all_possible_actions, get_random_possible_action


class MyPlayer(Player):
    def __init__(self, player_index=0) -> None:
        super().__init__()
        self.index = player_index
        self.steps = {}
        self.last_actions = []

    def print_steps(self):
        for key, value in self.steps.items():
            print(key, value)

    def update_scores(self, haswon: bool) -> None:
        if haswon:
            for key_board, key_action in self.last_actions:
                self.steps[key_board][key_action] += 1
        else:
            for key_board, key_action in self.last_actions:
                self.steps[key_board][key_action] -= 1

        self.last_actions = []

    def action_to_str(self, action: tuple[tuple[int, int], Move]) -> str:
        return str(action)

    def str_to_action(self, action_str: str) -> tuple[tuple[int, int], Move]:
        # ((0, 0), <Move.TOP: 0>)

        from_pos_str = action_str[1:7]  # "(0, 0)"
        move_index = action_str[-3]  # "0"

        from_pos = eval(from_pos_str)
        move = Move(int(move_index))
        return from_pos, move

    def board_to_str(self, board: list[list[int]]) -> str:
        return "".join([str(x) for row in board for x in row])

    def random_action(self, game: "Game") -> tuple[tuple[int, int], Move]:
        coors, move = get_random_possible_action(game, self.index)
        action = ((coors[1], coors[0]), move)
        return action

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        board = game._board
        action = self.random_action(game)

        key_board = self.board_to_str(board)
        key_action = self.action_to_str(action)

        if key_board not in self.steps:
            self.steps[key_board] = {}
            self.steps[key_board][key_action] = 0
            return action

        if key_action not in self.steps[key_board]:
            self.steps[key_board][key_action] = 0

        if key_board in self.steps and len(self.steps[key_board]) > 0:
            possible_actions = self.steps[key_board]
            action = max(possible_actions, key=possible_actions.get)
            action = self.str_to_action(action)

        self.last_actions.append((key_board, key_action))

        return action

    def train(self, iterations=100) -> None:
        print("Training...")
        wins = 0
        for _ in tqdm(range(iterations)):
            g = Game()
            winner = g.play(self, RandomPlayer())
            self.update_scores(winner == 0)
            wins += 1 if winner == 0 else 0
        print("Training done - Win rate:", wins / iterations)

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.steps, f)

    def load(self, filename: str) -> None:
        with open(filename, "rb") as f:
            self.steps = pickle.load(f)


if __name__ == "__main__":
    player1 = MyPlayer(player_index=0)
    player2 = RandomPlayer()

    # player1.train(20000)
    # player1.save("quixo/myplayer.pickle")
    player1.load("quixo/myplayer.pickle")
    # player1.print_steps()

    wins = 0
    for _ in tqdm(range(100)):
        g = Game()
        winner = g.play(player1, player2)
        wins += 1 if winner == 0 else 0
    print("Win rate:", wins / 100)

    # action = ((0, 0), Move.TOP)
    # print(action)
    # action_str = player1.action_to_str(action)
    # print(type(action_str))
    # action_back = player1.str_to_action(action_str)
    # print(action_back)
    # print(type(action_back[1]))
