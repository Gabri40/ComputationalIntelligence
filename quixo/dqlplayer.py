from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from game import Game, Move, Player
from collections import namedtuple
from tqdm import tqdm
from collections import deque


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


# =================================================================================================


CELLS_TAKEABILITY_MATRIX = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

FLATTENED_CELLS_TAKEABILITY_MATRIX = [
    x for row in CELLS_TAKEABILITY_MATRIX for x in row
]


# =================================================================================================


def board_to_neutral_board(board: list[list[int]]) -> list[list[int]]:
    return [[-1 if x == 1 else 1 if x == 0 else 0 for x in row] for row in board]


def invert_neutral_board(board: list[list[int]]) -> list[list[int]]:
    return [[-x for x in row] for row in board]


def flatten_board(board: list[list[int]]) -> list[int]:
    return [x for row in board for x in row]


def board_to_board_string(board: list[list[int]]) -> str:
    return "".join([str(x) for row in board for x in row])


def board_string_to_board(state: str) -> list[list[int]]:
    return [[int(state[i * 5 + j]) for j in range(5)] for i in range(5)]


def action_to_action_string(action: tuple[tuple[int, int], Move]) -> str:
    return f"{action[0][0]}{action[0][1]}{action[1].value}"


def action_string_to_action(action_string: str) -> tuple[tuple[int, int], Move]:
    return ((int(action_string[0]), int(action_string[1])), Move(int(action_string[2])))


# =================================================================================================

Experience = namedtuple(
    "Experience", ("player", "state", "action", "next_state", "reward")
)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def add(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(self.buffer, batch_size)


# =================================================================================================


class DQN(nn.Module):
    def __init__(self, input_size=26, output_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def interpret_output(self, q_values):
        # Reshape output to reflect the board structure and possible actions
        return q_values.view((5, 5, 4))

    def map_indices_to_actions(self, action_indices):
        actions = []
        for i in range(5):
            for j in range(5):
                cell_action = action_indices[i, j].item()
                if cell_action == Move.RIGHT.value:
                    actions.append(((i, j), Move.RIGHT))
                elif cell_action == Move.LEFT.value:
                    actions.append(((i, j), Move.LEFT))
                elif cell_action == Move.BOTTOM.value:
                    actions.append(((i, j), Move.BOTTOM))
                elif cell_action == Move.TOP.value:
                    actions.append(((i, j), Move.TOP))
        return actions

    def select_single_action(self, q_values):
        # Select the action with the highest Q-value for the entire move
        selected_action_index = np.unravel_index(
            np.argmax(q_values.numpy()), q_values.shape
        )
        selected_action = (
            (selected_action_index[0], selected_action_index[1]),
            Move(selected_action_index[2]),
        )
        return selected_action

    def save_weights(self, file_path="quixo/dql_weights.pt"):
        torch.save(self.state_dict(), file_path)
        print(f"Model weights saved to {file_path}")

    def load_weights(self, file_path="quixo/dql_weights.pt"):
        self.load_state_dict(torch.load(file_path))
        print(f"Model weights loaded from {file_path}")


# =================================================================================================
class DQNPlayer(Player):
    def __init__(self, dqn: DQN, target_dqn: DQN, player_index) -> None:
        super().__init__()
        self.dqn = dqn
        self.target_dqn = target_dqn
        self.replay_buffer = ReplayBuffer()
        self.game_memory = []
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.player_index = player_index

    def train(self, num_games: int, gamma: float):
        for i in tqdm(range(num_games)):
            game = Game()
            self.game_memory = []

            winner = game.play(self, RandomPlayer())

            # outcome
            reward = (
                1
                if self.player_index == winner
                else 0
                if winner == 1 - self.player_index
                else 0.5  # draw
            )

            # single reward update for all moves taken in single game and adds to reply buffer
            for exp in self.game_memory:
                exp.reward = reward if exp.player == self.player_index else -reward
                # self.replay_buffer.add(exp)

            # add to replay buffer with format compatible with nn input
            for exp in self.game_memory:
                board = flatten_board(exp.state)
                next_board = flatten_board(exp.next_state)
                self.replay_buffer.add(
                    Experience(
                        exp.player,
                        board,
                        exp.action,
                        next_board,
                        exp.reward,
                    )
                )

            # perform nn update for whole game
            batch = self.replay_buffer.sample(32)
            (
                batch_players,
                batch_states,
                batch_actions,
                batch_next_state,
                batch_rewards,
            ) = zip(*batch)

            batch_states = torch.tensor(batch_states, dtype=torch.float32)

            q_values = self.dqn(batch_states)

        return

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        # Get the current state of the board
        board = game.get_board()
        nn_input = flatten_board(board).append(self.player_index)

        # eps greedy
        if random.random() < 0.1:
            action = get_random_possible_action(game)
        else:
            # Get the Q-values for the current state
            q_values = self.dqn(nn_input)

            # Interpret the Q-values
            q_values = self.dqn.interpret_output(q_values)

            # Get the possible actions for the current state
            possible_actions = game.get_possible_actions()

            # Filter the Q-values to only include the possible actions
            q_values = q_values * torch.tensor(
                flatten_board(possible_actions), dtype=torch.float32
            )

            # Select the action with the highest Q-value
            action = self.dqn.select_single_action(q_values)

        # next board
        next_board = try_move(action[0], action[1], board, self.player_index)
        next_board_eval = evaluate_winner(next_board)
        next_board_eval = (
            1
            if self.player_index == next_board_eval
            else -1
            if next_board_eval == 1 - self.player_index
            else 0.5
        )

        exp = Experience(self.player_index, board, action, next_board, None)

        self.game_memory.append(exp)

        return (action[0][1], action[0][1]), action[1]


# =================================================================================================
# TRAINING
# =================================================================================================

# inputs
# baord flattened + one element for player -> 26

# outputs
# q values for each action -> 5*5*4

# model
dqn = DQN(26, 100)
target_dqn = DQN(26, 100)

# training
dqn_player = DQNPlayer(dqn, target_dqn, 0)
dqn_player.train(100, 0.9)

# save weights
dqn_player.dqn.save_weights()
