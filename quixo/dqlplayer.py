import utils
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


def board_to_neutral_flatten_board(board: list[list[int]], p_index=int) -> list[int]:
    """Converts a board to a neutral flatten board where current player is 1, opponent is -1 and empty is 0"""
    opponent = 1 - p_index
    return [
        1 if x == p_index else -1 if x == opponent else 0 for row in board for x in row
    ]


def action_to_triplet(action: tuple[int, int, int]) -> tuple[int, int, int]:
    """Converts an action to a tuple of int"""
    return (action[0][0], action[0][1], action[1].value)


def triplet_to_action(triplet: tuple[int, int, int]) -> tuple[tuple[int, int], Move]:
    """Converts a tuple of int to an action"""
    return ((triplet[0], triplet[1]), Move(triplet[2]))


def all_possible_actions_to_list_triplets(
    game: "Game", p_index
) -> list[tuple[int, int, int]]:
    """Converts a list of possible actions for the current to a list of tuple of int"""
    actions = utils.get_all_possible_actions(game.get_board(), p_index)
    return [(action[0][0], action[0][1], action[1].value) for action in actions]


def is_action_possible(
    game: "Game", p_index: int, action: tuple[int, int, int]
) -> bool:
    """Check if an action is possible"""
    actions = all_possible_actions_to_list_triplets(game, p_index)
    return action in actions


def only_possible_actions(
    game: "Game", p_index: int, actions: list[tuple[int, int, int]]
) -> list[tuple[int, int, int]]:
    """Return only the possible actions from a list of actions"""
    possible_actions = all_possible_actions_to_list_triplets(game, p_index)
    return [action for action in actions if action in possible_actions]


def reward_to_neutral_reward(reward: int, p_index: int) -> int:
    """Converts a reward to a neutral reward where current player is 1, opponent is -1 else 0"""
    if reward == p_index:
        return 1
    elif reward == 1 - p_index:
        return -1
    else:
        return 0


Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class MemoryRandomPlayer(Player):
    def __init__(self, p_index) -> None:
        super().__init__()
        self.p_index = p_index
        self.mem = []

    def get_memories(self) -> list[Experience]:
        return self.mem

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        from_pos, move = utils.get_random_possible_action(game, self.p_index)

        # experience
        state = board_to_neutral_flatten_board(game.get_board(), self.p_index)
        action = action_to_triplet((from_pos, move))
        tried_board = utils.try_move((from_pos, move), game.get_board(), self.p_index)
        evaluated_tried_board = utils.evaluate_winner(tried_board)
        next_state = board_to_neutral_flatten_board(tried_board, self.p_index)
        # reward = reward_to_neutral_reward(evaluated_tried_board, self.p_index)

        self.mem.append(Experience(state, action, next_state, None))

        return (from_pos[1], from_pos[0]), move  # GOD WHY


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQLPlayer(Player):
    def __init__(self, p_index) -> None:
        super().__init__()
        self.p_index = p_index
        self.model = DQN(25, 4 * 5 * 5)
        # 25 * 4actions, needs a mask
        # should return 4 5x5 matrices each matrix is q values for each move

    def get_memories(self) -> list[Experience]:
        return self.replay_buffer.buffer

    def get_model_output(self, game) -> torch.Tensor:
        state = board_to_neutral_flatten_board(game.get_board(), self.p_index)
        return self.model(torch.tensor(state, dtype=torch.float32))

    def model_output_view(self, game) -> torch.Tensor:
        return self.get_model_output(game).view(4, 5, 5)

    def mask_model_output(self, game: "Game") -> torch.Tensor:
        """Mask the model output to only possible actions"""
        model_output = self.get_model_output(game)
        model_output = model_output.view(4, 5, 5)

        possible_actions = all_possible_actions_to_list_triplets(game, self.p_index)
        possible_actions_to_model_output_mask = torch.zeros(4, 5, 5)

        for action in possible_actions:
            possible_actions_to_model_output_mask[action[2], action[0], action[1]] = 1

        return model_output * possible_actions_to_model_output_mask

    def get_max_q_action(self, game: "Game") -> tuple[int, int, int]:
        """Returns indexes of the action with the highest q value"""
        masked_model_output = self.mask_model_output(game)
        # masked 4 * [5 * 5] matrix

        # Find the index of the maximum value across all matrices
        flat_index = torch.argmax(masked_model_output.view(-1))

        # Calculate matrix, row, and column indices directly
        matrix_index = flat_index // (
            masked_model_output.shape[1] * masked_model_output.shape[2]
        )
        row_index = (
            flat_index % (masked_model_output.shape[1] * masked_model_output.shape[2])
        ) // masked_model_output.shape[2]
        col_index = (
            flat_index % (masked_model_output.shape[1] * masked_model_output.shape[2])
        ) % masked_model_output.shape[2]

        return (row_index.item(), col_index.item(), matrix_index.item())

    def action_to_move(
        self, action: tuple[int, int, int]
    ) -> tuple[tuple[int, int], Move]:
        """Converts an action triplet to a move"""
        return ((action[0], action[1]), Move(action[2]))

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        if random.random() < 0:
            action = utils.get_random_possible_action(game, self.p_index)
        else:
            action = self.get_max_q_action(game)
            action = self.action_to_move(action)

        action = (action[0][1], action[0][0]), action[1]  # GOD WHY
        print(action)
        return action


if __name__ == "__main__":
    game = Game()
    p1 = DQLPlayer(0)
    p2 = RandomPlayer()
    res = game.play(p1, p2)
    print(res)
