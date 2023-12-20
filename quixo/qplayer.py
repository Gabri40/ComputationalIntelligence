import random
from game import Game, Move, Player
from tqdm import tqdm
from utils import get_all_possible_actions, get_random_possible_action
import numpy as np

import pickle


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class QPlayer(Player):
    def __init__(
        self, player_index=0, preload=False, random_action_probability=0.1
    ) -> None:
        super().__init__()
        self.player_index = player_index
        self.opponent_index = 1 - player_index
        self.q_table = {}
        self.random_action_probability = random_action_probability
        self.last_game_actions = []

        if preload:
            try:
                print("loading q table (can take a bit)...")
                with open("quixo/qplayer.pickle", "rb") as f:
                    self.q_table = pickle.load(f)
            except:
                raise Exception("no q table found")
            else:
                print("q table loaded")

    def save_q_table(self):
        print("saving q table (can take a bit)...")
        with open("quixo/qplayer.pickle", "wb") as f:
            pickle.dump(self.q_table, f, protocol=5)

    def board_to_key(self, board: list[list[int]]) -> str:
        """
        transform the board into unique string by concat all the values

        - player1 has index 0
        - player2 has index 1

        example:
            empty board -> '-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1'
        """
        # need to make the key agnostic to the player index so we can use it for
        # both players

        # can train the player as player 0 if the player is player 1, we need to
        # invert the board => just invert the key ???

        board_str = "".join([str(x) for row in board for x in row])

        if self.player_index == 1:
            board_str = board_str.replace("0", "1").replace("1", "0")

        return board_str

    def move_to_key(self, action: tuple[tuple[int, int], Move]) -> str:
        """
        transform the move into unique string by concat all the values

        example: # ((0, 0), Move.TOP) -> '000'
        """
        row = action[0][0]
        col = action[0][1]
        move_value = action[1].value
        return str(row) + str(col) + str(move_value)

    def movekey_to_move(self, movekey: str) -> tuple[tuple[int, int], Move]:
        """
        transform the movekey string into move

        example: '000' -> ((0, 0), Move.TOP)
        """
        row = int(movekey[0])
        col = int(movekey[1])
        move_value = int(movekey[2])
        return ((row, col), Move(move_value))

    def add_new_board(self, game) -> None:
        """
        add new board and all possible actions to the q table
        """
        board_key = self.board_to_key(game._board)
        all_action_keys = [
            self.move_to_key(action)
            for action in get_all_possible_actions(game, self.player_index)
        ]

        self.q_table[board_key] = {}
        for action_key in all_action_keys:
            self.q_table[board_key][action_key] = 0

    def get_max_q_value_move(self, game: "Game") -> float:
        """
        return move with max q value for the current board
        """
        board_key = self.board_to_key(game._board)
        max_q_value = max(self.q_table[board_key].values())
        max_q_value_moves = [
            self.movekey_to_move(movekey)
            for movekey, q_value in self.q_table[board_key].items()
            if q_value == max_q_value
        ]
        return random.choice(max_q_value_moves)

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        """
        exploration vs exploitation

        chance to make a random move or the move with max q value
        """
        current_board_key = self.board_to_key(game._board)
        if current_board_key not in self.q_table:
            self.add_new_board(game)

        action = None
        if random.random() < self.random_action_probability:
            action = get_random_possible_action(game, self.player_index)
        else:
            action = self.get_max_q_value_move(game)

        action_key = self.move_to_key(action)

        if action_key not in self.q_table[current_board_key]:
            raise Exception("action key not in q table but it shuld be")

        self.last_game_actions.append((current_board_key, action_key))

        return (action[0][1], action[0][0]), action[1]  # hate it

    def update_q_table(self, haswon: bool) -> None:
        """
        update q table after the game is finished
        """
        if haswon:
            for board_key, action_key in self.last_game_actions:
                self.q_table[board_key][action_key] += 1
        else:
            for board_key, action_key in self.last_game_actions:
                self.q_table[board_key][action_key] -= 1

        self.last_game_actions = []

    def train(self, iterations=1000) -> None:
        """
        train the player against a random player
        """
        for _ in tqdm(range(iterations)):
            game = Game()
            winner = game.play(self, RandomPlayer())
            self.update_q_table(winner == self.player_index)

        try:
            self.save_q_table()
        except:
            raise Exception("could not save q table")
        else:
            print("q table saved")


if __name__ == "__main__":
    player_index = 0
    training = True
    training = False  # comment out for training

    if training:
        print("training...")
        player_index = 0
        # false trains from scratch, true loads the q table and continues training
        player = QPlayer(player_index=player_index, preload=True)
        random_player = RandomPlayer()
        # player.train(iterations=1000)
        player.train(iterations=10000)
        # player.train(iterations=100000)

        # oh god its 1GB after 2 100k iterations

    else:
        print("playing...")
        player = QPlayer(player_index=player_index, preload=True)
        wins = 0
        games = 1000
        for _ in tqdm(range(games)):
            game = Game()
            if player_index == 0:
                winner = game.play(player, RandomPlayer())
            else:
                winner = game.play(RandomPlayer(), player)
            if winner == player_index:
                wins += 1
        print(f"win rate: {wins/games}")


# SOME RESULTS

# this thing is too big maybe deep QL


# TRAINING FROM SCRATCH

# trainging iterations = 1000, random_action_probability = 0.1

# player pos = 0
#    100%|██████████████████████████| 1000/1000 [00:05<00:00, 194.04it/s]
#    win rate: 0.514

# player pos = 1
#   100%|██████████████████████████| 1000/1000 [00:04<00:00, 203.32it/s]
#   win rate: 0.504


# trainging iterations = 10000, random_action_probability = 0.1

# player pos = 0
#   100%|██████████████████████████| 1000/1000 [00:04<00:00, 203.04it/s]
#   win rate: 0.546

# player pos = 1
#   100%|██████████████████████████| 1000/1000 [00:04<00:00, 202.53it/s]
#   win rate: 0.491


# trainging iterations = 100000, random_action_probability = 0.1

# player pos = 0
#   100%|██████████████████████████| 1000/1000 [00:02<00:00, 337.90it/s]
#   win rate: 0.561

# player pos = 1
#   100%|██████████████████████████| 1000/1000 [00:02<00:00, 340.40it/s]
#   win rate: 0.519
