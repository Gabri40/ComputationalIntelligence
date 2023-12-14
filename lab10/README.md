# Lab10.ipynb

Implementation of a Q-Learning agent that is trained and tested against different types of opponents in a game environment.
The opponents are:

- a random player
- a minimax player
- another Q-Learning agent

### Man vs Machine

At the bottom of the file a section to play against each agent and teacher is available.

## Training

The Q-Learning agent is trained using the `train(agent, opponent, num_episodes)` function.

The agent and opponent are passed as parameters, along with the number of episodes to train for. The function will return a Q-table that can be used to test the agent against the opponent.

The `.pkl` are obtained by training each agent for 100_000 episodes.

The Q-tables are saved to a `.pkl` file using the `save_q_table(q_table, filename)` function.

## Testing

Each agent trained in the previous section is tested against the other opponents using the `test(agent, opponent)` function. The function displays the results of the matchup.

As expected the minimax agent is the strongest, followed by the Q-Learning agents and finally the random agent.

Results:

```
RANDOM TRAINED AGENT VS RANDOM
 52 wins
 15 ties
 33 losses
```

```
MINIMAX TRAINED AGENT VS MINIMAX
 10 wins
 3 ties
 87 losses
```

```
SELF TRAINED AGENT VS SELF
 65 wins
 11 ties
 24 losses
```

```
RANDOM TRAINED AGENT VS MINIMAX
 8 wins
 0 ties
 92 losses
```

```
RANDOM TRAINED AGENT VS SELF
 55 wins
 16 ties
 29 losses
```

```
MINIMAX TRAINED AGENT VS RANDOM
 60 wins
 11 ties
 29 losses
```

```
MINIMAX TRAINED AGENT VS SELF
 57 wins
 16 ties
 27 losses
```

```
SELF TRAINED AGENT VS RANDOM
 54 wins
 11 ties
 35 losses
```

```
SELF TRAINED AGENT VS MINIMAX
 11 wins
 5 ties
 84 losses
```
