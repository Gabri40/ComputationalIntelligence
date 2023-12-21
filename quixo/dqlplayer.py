from utils import *
import torch
import numpy as np
import random
from game import Game, Move, Player
from collections import namedtuple
from tqdm import tqdm


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class DQN(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


Experience = namedtuple(
    "Experience", field_names=["state", "action", "next_state", "reward"]
)


class DQNPlayer(Player):
    def __init__(self, player_index, epsilon=0.1, gamma=0.9, learning_rate=0.001):
        super().__init__()
        self.player_index = player_index
        self.epsilon = epsilon
        self.gamma = gamma

        # memory
        self.memory = []
        self.memory_size = 10000

        # nn
        self.neural_network = DQN(25, 4)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.neural_network.parameters(), lr=learning_rate
        )

        # loss
        self.loss = torch.nn.MSELoss()

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        state = game.get_board()

        # Select an action using epsilon-greedy strategy
        epsilon = 0.1
        if random.random() < epsilon:
            # Explore: Select a random move
            return get_random_possible_action(game, self.player_index)

        # Exploit: Select the move with the highest Q-value
        with torch.no_grad():
            input_tensor = torch.tensor(
                [c for row in state for c in row], dtype=torch.float32
            )
            q_values = self.neural_network(input_tensor)

        # Find the move with the highest Q-value
        best_move_index = torch.argmax(q_values).item()
        # Convert the move index back to (row, col, slide)
        num_row_col_possibilities = 5
        num_slide_directions = 4
        chosen_row = best_move_index // num_row_col_possibilities
        chosen_col = best_move_index % num_row_col_possibilities
        chosen_slide = best_move_index % num_slide_directions
        move = (chosen_row, chosen_col), Move(chosen_slide)

        # try the move
        next_state, acceptable = try_move(move[0], move[1], game, self.player_index)
        penalty = 0.1 if not acceptable else 0
        next_state_eval = evaluate_board(next_state)
        if next_state_eval == self.player_index:
            reward = 1
        elif next_state_eval == -1:
            reward = 0
        else:
            reward = -1
        reward -= penalty

        # Store the experience in memory
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        experience = Experience(state, move, next_state, reward)
        self.memory.append(experience)

        # Update the Q-values using the Bellman equation
        self.update_q_values()

        return move

    def update_q_values(self):
        # If memory is not full, do nothing
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences
        # Sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states = [s for s, _, _, _ in batch]
        actions = [(a[0][0], a[0][1], a[1].value()) for _, a, _, _ in batch]
        next_states = [ns for _, _, ns, _ in batch]
        rewards = [r for _, _, _, r in batch]

        # Check if actions is iterable
        if not isinstance(actions, (list, tuple)):
            raise TypeError(
                f"Expected actions to be a list or tuple, but got {type(actions).__name__}"
            )

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        # Compute Q-values
        q_values = self.neural_network(states)
        next_q_values = self.neural_network(next_states)
        next_q_values = next_q_values.detach()
        best_next_q_values = next_q_values.max(dim=1)[0]
        q_values = q_values.gather(index=actions.unsqueeze(-1), dim=1).squeeze(-1)

        # Compute the expected Q-values
        expected_q_values = rewards + self.gamma * best_next_q_values

        # Compute loss
        loss = self.loss(q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes: int, batch_size: int):
        print("Training")
        self.batch_size = batch_size
        for episode in tqdm(range(num_episodes)):
            # Play a game
            game = Game()
            game.play(self, RandomPlayer())

            # Update the Q-values
            self.update_q_values()

            # Save the model
            if episode % 100 == 0:
                torch.save(self.neural_network.state_dict(), "model.pth")


if __name__ == "__main__":
    player = DQNPlayer(0)
    player.train(1000, 32)

    wins = 0
    for i in range(100):
        game = Game()
        winner = game.play(player, RandomPlayer())
        if winner == 0:
            wins += 1
    print(wins)
