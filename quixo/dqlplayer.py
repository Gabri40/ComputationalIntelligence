import random
from game import Game, Move, Player
from tqdm import tqdm
from game import Game, Move
from utils import get_all_possible_actions, get_random_possible_action
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


class DQLPlayer(Player):
    def __init__(self, player_index, neural_network) -> None:
        super().__init__()
        self.player_index = player_index
        self.opponent_index = 1 - player_index
        self.neural_network = DQN(input_size, output_size)
        self.optimizer = torch.optim.Adam(neural_network.parameters(), lr=0.01)
        self.loss_function = torch.nn.MSELoss()

    def board_to_net_input(self, game: "Game"):
        board = np.array(game._board).flatten()
        board_tensor = torch.tensor(board, dtype=torch.float32)
        return board_tensor

    def net_output_to_move(self, output):
        # Decode the chosen move
        num_row_col_possibilities = 5
        num_slide_directions = 4

        # Map the output values to the corresponding indices
        chosen_row = output[0].item() * num_row_col_possibilities
        chosen_col = output[1].item() * num_row_col_possibilities
        chosen_slide = output[2].item() * num_slide_directions

        # Round the values to the nearest integer
        chosen_row = round(chosen_row)
        chosen_col = round(chosen_col)
        chosen_slide = round(chosen_slide)

        print(chosen_row, chosen_col, chosen_slide)
        return chosen_row, chosen_col, chosen_slide

    def select_move(self, game):
        input_tensor = self.board_to_net_input(game)
        output = self.neural_network(input_tensor)
        chosen_move = self.net_output_to_move(output)
        return chosen_move

    def make_move(self, game: Game):
        move = self.select_move(game)
        move = (move[1], move[0]), Move(move[2])

        if move not in get_all_possible_actions(game, self.player_index):
            penalty = -0.1

        return move

    def receive_reward(self, winner_index):
        if winner_index == self.player_index:
            reward = 1
        elif winner_index == self.opponent_index:
            reward = -1
        else:
            reward = 0

        return reward

    def train(self, num_episodes, batch_size):
        random_agent = RandomPlayer()
        for episode in range(num_episodes):
            game = Game()
            winner = game.play(self, random_agent)

            reward = self.receive_reward(winner)

            self.optimizer.zero_grad()
            loss = self.compute_loss(reward)
            print(f"episode {episode}: loss -> {loss}")
            loss.backward()
            self.optimizer.step()

    def compute_loss(self, reward):
        # mse loss
        return


if __name__ == "__main__":
    # neural network, optimizer, and loss function
    input_size = 25
    output_size = 3

    model_path = "trained_model.pth"

    dql_player = DQLPlayer(0, neural_network, optimizer, loss_function)
    num_episodes = 1
    batch_size = 32
    # saving
    dql_player.train(num_episodes, batch_size)
    torch.save(dql_player.neural_network.state_dict(), model_path)

    # loading
    neural_network = DQN(input_size, output_size)
    neural_network.load_state_dict(torch.load(model_path))

    dql_player = DQLPlayer(0, neural_network, optimizer, loss_function)
    random_agent = RandomPlayer()

    wins = 0
    games = 100
    for _ in tqdm(range(games)):
        game = Game()
        winner = game.play(dql_player, random_agent)
        wins += winner == 0
    print(f"Win rate against random agent: {wins/games}")

# 100 training episodes -> 0.2 win rate
# 1000 training episodes -> 0.39 win rate
