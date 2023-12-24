# Computational Intelligence Log

commits will be added later since already tracked by git

# LAB2 - Nim Peer Reviews - 24/11/2023

### https://github.com/FedeBucce/Computational_intelligence/

    **Positive Aspects**

    Modularity and Readability:
    - Use of well-defined functions and classes improves code organization.
    - The Individual class encapsulates relevant information

    Evolutionary Strategy Implementation:
    - The implementation of the (1,λ) Evolutionary Strategy is clear and follows a standard structure.

    Overall the lab seems well implemented 😄

    **Negative Aspects**

    Evaluation:
    - The agent using adaptive strategies is always player 0, giving it and advantage. Starting player could be random for a more fair evaluation.

### https://github.com/AllegraRoberto/Computational-Intelligence/

    **Positive Aspects**

    Documentation:
    - Very clear and extensive README allows for clear understanding of the implementation

    Modularity and Readability:
    - Well-defined functions and comments improves organization and readability

    Evaluation:
    - Extensive evaluation of the best candidates and comparisons of the results between generations allow for a good analysis if the ES used.

    Overall the lab seems well implemented 😄

    **Negative Aspects**

    Code Repetition:
    - Some code within the simulation function is repeated often, could be clearer with some refactoring.

    Evaluation:
    - The agent using adaptive strategies is always player 0, giving it and advantage. Starting player could be random for a more fair evaluation.

# LAB9 - Peer Reviews - 07/12/2023

### https://github.com/DonatoLanzillotti/Computational_Intelligence23

    **Overall Structure and Organization**
    The code is well structured and organized using classes for different components of the evolutionary algorithm. This makes the code more readable and easy to understand. The plots for the fitness are a nice addition but the readme could be a bit more extensive.😊

    **EA Implementation**
    The EA algorithm has a standard structure with some good improvement like the check of saturation and the addiction of the already evaluated genomes dict to save computation.

    The increase in mutation probability seems to me a bit steep:

        if cnt % 25 == 0 and cnt > 0:
                        self.mutation_prob = self.mutation_prob * 1.25

    Given the starting point of .35 and 1.25 multiplayer the probability surpasses 1 after just 5 updates, but this was maybe by design.

    **Final Comments**
    Overall the code and the results look good to me!😄

### https://github.com/ahmadrezafrh/Computational-Intelligence

    **Organization and Structure**
    No readme. The code itself is clear but some comments would help a first glance understanding of what each component does. The the graphs showing the fitness evolution are a nice addition. 😊

    **Algorithm Implementation**
    The algorithm implements correctly a normal EA with some nice additions:
    - Switching between mutation and crossover mode if the algo reaches a "patience" limit is a good idea and seems to provide a good exploration/exploitation balance
    - Early termination based on the amount of the above switches to mutation

    The actual algorithm could be implement as a function to avoid cells repetition.

    **Final Comments**
    Aside some minor organization considerations the code and results look good to me.😄

# LAB10 - Peer Reviews -

TODO after Christmas

# QUIXO

### 1. Introduction

The goal of this project is to implement a agent for the game of Quixo.

I tried different approaches, from minimax to a Q-Learning player, ending with a player using Deep Q-Learning with a replay buffer.

Most of the approaches i tried are in the `quixo` folder aside the very catastrophic ones .

# 2. The Players

## 2.1 Random Player

Was already implemented in the lab, was used to test the other players.

## 2.2 Heuristic Player Player

First attempt at implementing an agent based on some simple heuristic after reading the rules of the game and looking at some gameplay online.

This was really bad and its not commited to the repo.

**Main ideas:**

- At each turn try all the possible moves ( this was achieved by by using a copy and paste version of the \_\_move, \_\_take, \_\_slide as well as a function to find all the possible moves for a certain player)

- After each try evaluate the board with a simple heuristic function :

```python
def heuristic(self, board, player):
        def count_lines(board, player):
            # Count the number of lines with 4 pieces of the player ie 4 tiles with value =self.index
            count = 0
            for i in range(5):
                if np.sum(board[i:] == player) >= 4:
                    count += 1
                if np.sum(board[:, i] == player) >= 4:
                    count += 1
                if np.sum(np.diag(board, i) == player) >= 4:
                    count += 1
                if np.sum(np.diag(board[:, ::-1], i) == player) >= 4:
                    count += 1
            return count

        def count_corners(board, player):
            # Count the number of corners controlled by the player
            corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
            count = sum(1 for corner in corners if board[corner] == player)
            return count

        def center_control(board, player):
            # Check if the player controls the center of the board
            center = (2, 2)
            return 1 if board[center] == player else 0

        def edge_control(board, player):
            # Count the number of edges controlled by the player
            edges = [(0, 2), (2, 0), (2, 4), (4, 2)]
            count = sum(1 for edge in edges if board[edge] == player)
            return count

        # Calculate the score based on different features
        weights = self.weights
        player = self.index
        score = (
            weights["num_lines"] * count_lines(board, player)
            + weights["num_corners"] * count_corners(board, player)
            + weights["center_control"] * center_control(board, player)
            + weights["edge_control"] * edge_control(board, player)
        )
        return score
```

**Issues:**
I dont think the game of Quixo is suited for a heuristic approach (or rather i just couldn't define a good one): the number of possible moves is high and with the board changing drastically at each turn i found it hard to define a good heuristic.

The result were indistinguishable from just choosing a random action.

**Takeaways:**

- Using a copy and paste version of the \_\_move, \_\_take, \_\_slide and most imporantly a function to find all the possible moves for a certain player is a was a good idea and i brought them over to other approaches with refinements.

## 2.3 Q-Learning Player

After the failure of the heuristic approach i decided to try a reinforcement learning approach after looking at some inspiration online. First time trying to implement a RL algorithm so i was a bit lost at the beginning.
Went over various iterations and rewrites befire getting an understanding of what i was doing adn getting a working player.

**Main ideas:**

- Implementing the Q table as a dictionary with the state as key and the value as the Q value for each action.

- Before the state and actions are added to the Q table they are transformed to strings (immutable) to be used as key, making sure to tranform the board in a way that is agnostic to the player index (so that the same board is always represented in the same way).

  ```python
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
  ```

- Epson greedy approach is used to choose the action to take

  ```python
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

          return (action[0][1], action[0][0]), action[1]
  ```

- The Q values are updated at the end of each game based of the win or lose.

  ```python
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
  ```

**Issues:**

While i can see it working with enough compute the main issue is the size of the Q table: the number of possible states is huge and the Q table grows exponentially with the number of games played.

After training for 100_000 games the Q table was already 400MB and the player while improving was still not much better than a random player.

Q table is not commited for size reasons.

**Takeaways:**

- Most important takeaway is making the board agnostic to the player index which i didn't do before as well as converting the board adn actions to a more compact representation (will be useful later when finally implementing DQL).

- The Q table was a good idea but the size of it is a problem.

- I created an `utils.py` file to store all the functions like getting all the possible actions and trying a move.

## 2.4 Minimax Player

Tried my hand at a minimax player using alpha beta pruning. Like with the heuristic player i found it hard to define a good heuristic for the game of Quixo.

**Main ideas:**

- Standard Minimax with alpha beta pruning

- Using the same functions as before to get all the possible moves and try them while evaluating with a "new" heuristic function

  ```python
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
  ```

**Issues:**
While the algorithm works i still had issues trying to find a combination things to consider and the weights for them in the heuristic function that would make the player play well.

Again it was winning only slightly more than a random player.

**Takeaways:**
Doing some research online i found some informations:

- Quixo is a solved game ()[] but with exponential complexity ()[]. I couldn't however follow the implementation of the algorithm in the paper.

- The structure of the game can leading to repeating patterns and this can hinder performance of a minimax algorithm. I dont doubt hoever that my implementation of it and heuristic function were not good enough to reach this limitations. However after getting these informations i decided to move to Deep Q-Learning.

## 2.5 Deep Q-Learning Player

Final player implemented, it took some time to understand how to implement it and i had to rewrite it many times bifore getting an understanding of what i was doing.

The main struggle i faced was trying to understand how to read and evalualte the output of the network after giving it the current state as well as how to evaluate the loss and update the network.

The complete dql player code is also copied all at once at the end of the log.

**Main ideas:**

- Experience and Replay Buffer: at each step the player saves the current state, action, reward and next state in a replay buffer. This is used to train the network at the end of each game. The main ideas behind the Replay Buffer are the following:

  - The network is trained at the end of each game with a batch of random samples from the replay buffer. This is done to avoid the network overfitting on only the last moves of the game.
  - It helps mitigate instabilities and allow the network to learn from previous experiences.

  ```python
  Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))

  class ReplayBuffer:
      def __init__(self, max_size: int = 100_000) -> None:
          self.buffer = deque(maxlen=max_size)

      def add(self, experience: tuple) -> None:
          self.buffer.append(experience)

      def sample(self, batch_size: int) -> list:
          return random.sample(self.buffer, batch_size)

      def __len__(self) -> int:
          return len(self.buffer)
  ```

- Conversion of the board to a neutral representation agnostic to the player index as well as converting the actions to a more compact representation.

  ```python
  def board_to_neutral_flatten_board(board: list[list[int]], p_index=int) -> list[int]:
    """Converts a board to a neutral flatten board where current player is 1, opponent is -1 and empty is 0"""
    opponent = 1 - p_index
    return [
        1 if x == p_index else -1 if x == opponent else 0 for row in board for x in row
    ]

  def reward_to_neutral_reward(reward: int, p_index: int) -> int:
    """Converts a reward to a neutral reward where current player is 1, opponent is -1 else 0"""
    if reward == p_index:
    return 1
    elif reward == 1 - p_index:
    return -1
    else:
    return 0.5

  ```

- The model is a pytorch neural network with the following structure.

  - The input will be the flattened board converted to a neutral representation
  - The output will be a 4\*5\*5 tensor. So it will be viewed as a 4 5\*5 matrices. Each matrix corrispond to one of the move directions (TOP, BOTTOM, LEFT, RIGHT) and each cell of the matrix will contain the Q value for that move in that position.

  ```python
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
  ```

- The models output a Q value for every cell and for every move but not all of them are possible (either illegal or the board configuration doesn't allow it). To account for this the output tensor is first converted to the (4,5,5) view described above before being masked with a mask created from all the possible actions at the current turn. Then the indices of the action with the highest Q value are taken and converted to a valid move format. The whole process is done with the following functions inside the player class:

  ```python
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
  ```

- At each turn the move is chosen with an epsilon greedy approach and the experience is added to a current game buffer.

  ```python
  def add_experience(
      self,
      game: "Game",
      action: tuple[int, int, int],
  ) -> None:
      """Adds an experience to the memory"""
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
      return action
  ```

- For the training loop the theplayer plays a number of games against a random player and at the end of each game the current game buffer is added to the replay buffer. If the replay buffer has enough memories (to avoid overfitting the first matches) it is sampled and the network is updated on the samples using MSE loss and Adam optimizer.:

  ```python
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

              loss = self.loss(current_q_values, target_q_values)
              self.optimizer.zero_grad()
              loss.backward()
              self.optimizer.step()

          self.save()
  ```

**Issues:**
Many Deep Q Learnig implementations use a target network to avoid the network overfitting on the last moves of the game while this doesn't.

**Takeaways:**
The network is able to beat a random player more than 80% of the times on average both as first and second player.

    # DQNPlayer as P1 - Win rate over 1000 games: 0.838
    # DQNPlayer as P2 - Win rate over 1000 games: 0.797

    # DQNPlayer as P1 - Win rate over 1000 games: 0.836
    # DQNPlayer as P2 -  Win rate over 1000 games: 0.804

    # DQNPlayer as P1 - Win rate over 1000 games: 0.852
    # DQNPlayer as P2 -  Win rate over 1000 games: 0.806

These results are obtained after trainging the player for 1000 games with the replay buffer and the value of epsilon for the greedy choice is 0.1 .

Given the decisely worse performance of the other players i implemented i consider this a good result and don't think it is necessary to implement evaluation against them as well as implementing a target network.

This seems the best approach given both the results and the structure of the game itself.

# Final DQL Player

```python
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

        action = (action[0][1], action[0][0]), action[1]
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

            loss = self.loss(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.save()

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

```
