import random
from game import Game, Move, Player
from tqdm import tqdm
from dqlplayer import DQLPlayer


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


# class MyPlayer(Player):
#     def __init__(self) -> None:
#         super().__init__()

#     def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
#         from_pos = (random.randint(0, 4), random.randint(0, 4))
#         move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
#         return from_pos, move


if __name__ == "__main__":
    games = 1000
    player1 = DQLPlayer(0, preload=True)
    player2 = RandomPlayer()
    wins1 = 0
    for _ in tqdm(range(games)):
        g = Game()
        winner = g.play(player1, player2)
        wins1 += 1 if winner == 0 else 0

    player1 = RandomPlayer()
    player2 = DQLPlayer(1, preload=True)
    wins2 = 0
    for _ in tqdm(range(games)):
        g = Game()
        winner = g.play(player1, player2)
        wins2 += 1 if winner == 1 else 0

    print(f"\nDQNPlayer as P1 - Win rate over {games} games: {wins1 / games}")
    print(f"DQNPlayer as P2 -  Win rate over {games} games: {wins2 / games}")


# /Users/gabriquaranta/repos/computational-intelligence/.env
# CI/bin/python /Users/gabriquaranta/repos/computational-inte
# lligence/quixo/main.py
# 100%|█████████████████| 1000/1000 [00:04<00:00, 220.79it/s]
# 100%|█████████████████| 1000/1000 [00:04<00:00, 228.79it/s]

# DQNPlayer as P1 - Win rate over 1000 games: 0.838
# DQNPlayer as P2 -  Win rate over 1000 games: 0.797
