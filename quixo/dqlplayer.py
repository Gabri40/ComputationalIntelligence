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


def board_to_neutral_flatten_board(board: list[list[int]], p_index=int) -> list[int]:
    """Converts a board to a neutral flatten board where current player is 1, opponent is -1 and empty is 0"""
    opponent = 1 - p_index
    return [
        1 if x == p_index else -1 if x == opponent else 0 for row in board for x in row
    ]


def all_possible_actions_to_list_triplets(
    game: "Game", p_index
) -> list[tuple[int, int, int]]:
    """Converts a list of possible actions for the current to a list of tuple of int"""
    actions = utils.get_all_possible_actions(game.get_board(), p_index)
    return [(action[0][0], action[0][1], action[1].value) for action in actions]


def reward_to_neutral_reward(reward: int, p_index: int) -> int:
    """Converts a reward to a neutral reward where current player is 1, opponent is -1 else 0"""
    if reward == p_index:
        return 1
    elif reward == 1 - p_index:
        return -1
    else:
        return 0.5


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
    """
    Deep Q Learning Player.

    Parameters:

    - p_index: The player index.
    - preload: Whether to preload the model from a file. (default: False)

    Choose the action using epsilon greedy.

    The model is a DQN with 25 input neurons and 4 * 5 * 5 output neurons.

    The input neurons are the 25 board positions, 1 if the position is the
    current player, -1 if the position is the opponent, 0 otherwise.

    The output neurons are the 4 possible actions for each of the 5x5 board

    The model is trained using a replay buffer.
    """

    def __init__(self, p_index, preload=False) -> None:
        super().__init__()
        self.p_index = p_index
        self.model = DQN(25, 4 * 5 * 5)
        # 25 * 4actions, needs a mask
        # should return 4 5x5 matrices each matrix is q values for each move

        if preload:
            self.model.load_state_dict(torch.load("quixo/dqlmodel.pt"))

        self.one_game_memory = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss = nn.MSELoss()

    def save(self, path: str = "quixo/dqlmodel.pt") -> None:
        """Saves the model to a file"""
        torch.save(self.model.state_dict(), path)

    def get_memories(self) -> list[Experience]:
        """Returns the memories of the last game"""
        return self.one_game_memory

    def get_model_output(self, game) -> torch.Tensor:
        """Returns the model output for the current game board"""
        state = board_to_neutral_flatten_board(game.get_board(), self.p_index)
        return self.model(torch.tensor(state, dtype=torch.float32))

    def model_output_view(self, game) -> torch.Tensor:
        """Returns the model output for the current game board reshaped to 4 5x5 matrices one matrix per slide direction"""
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
        # masked 4 * [5 * 5] matrix
        masked_model_output = self.mask_model_output(game)

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

    def add_experience(
        self,
        game: "Game",
        action: tuple[int, int, int],
    ) -> None:
        """Adds an experience to the memory"""
        # Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))
        state = board_to_neutral_flatten_board(game.get_board(), self.p_index)
        next_state = board_to_neutral_flatten_board(
            utils.try_move(self.action_to_move(action), game.get_board(), self.p_index),
            self.p_index,
        )
        reward = reward_to_neutral_reward(
            utils.evaluate_winner(game.get_board()), self.p_index
        )
        experience = Experience(state, action, next_state, reward)

        self.one_game_memory.append(experience)

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        """Makes a move eps greedy"""
        if random.random() < 0.1:
            action = utils.get_random_possible_action(game, self.p_index)
        else:
            action = self.get_max_q_action(game)
            self.add_experience(game, action)
            action = self.action_to_move(action)

        action = (action[0][1], action[0][0]), action[1]  # GOD WHY
        # print(action)
        return action

    def train(self, num_games=1_000):
        """Trains the model"""
        print("Training...")
        buffer = ReplayBuffer()
        for g in tqdm(range(num_games)):
            game = Game()
            p1 = DQLPlayer(0)
            p2 = RandomPlayer()
            res = game.play(p1, p2)
            neutral_result = reward_to_neutral_reward(res, 0)

            for m in p1.get_memories():
                e = Experience(
                    m.state, m.action, m.next_state, m.reward - neutral_result
                )
                buffer.push(e)

            batch_size = 64
            if len(buffer) < batch_size:
                continue

            experiences = buffer.sample(batch_size)

            states = torch.tensor([e.state for e in experiences], dtype=torch.float32)
            actions = torch.tensor([e.action for e in experiences], dtype=torch.int64)
            next_states = torch.tensor(
                [e.next_state for e in experiences], dtype=torch.float32
            )
            rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)

            current_q_values = self.model(states).gather(1, actions).squeeze(-1)

            next_q_values = self.model(next_states).max(1)[0].detach()
            target_q_values = rewards + 0.99 * next_q_values
            target_q_values = target_q_values.unsqueeze(1)

            # print("\n\n")
            # print(current_q_values.shape, target_q_values.shape)
            # print(current_q_values.shape, target_q_values.shape)
            # print("\n\n")

            loss = self.loss(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.save()


# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING
# >.8 WIN RATE DO NOT TOUCH A SINGLE THING


if __name__ == "__main__":
    # player = DQLPlayer(0, preload=False)
    # player.train()

    games = 100
    wins = 0
    p1 = DQLPlayer(0, True)
    p2 = RandomPlayer()
    for _ in tqdm(range(games)):
        game = Game()
        res = game.play(p1, p2)
        if res == 0:
            wins += 1

    print(f"Win rate: {wins/games}")
