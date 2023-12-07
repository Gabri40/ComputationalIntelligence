import random
from game import Game, Move, Player
from player_heuristic import MyPlayer


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
    print("Random vs MyPlayer")
    s = 0
    for i in range(1000):
        g = Game()
        player1 = MyPlayer(1)
        player2 = RandomPlayer()
        winner = g.play(player2, player1)
        s += winner
    print(" MyPlayer wins:", s)

    print("\nMyPlayer vs Random")
    s = 0
    for i in range(1000):
        g = Game()
        player1 = MyPlayer(0)
        player2 = RandomPlayer()
        winner = g.play(player1, player2)
        s += 1 - winner
    print(" MyPlayer wins:", s)

    print("\nMyPlayer vs MyPlayer")
    s = 0
    p1, p2 = 0, 0
    for i in range(1000):
        g = Game()
        player1 = MyPlayer(0)
        player2 = MyPlayer(1)
        winner = g.play(player1, player2)
        p1 += 1 if winner == 0 else 0
        p2 += 1 if winner == 1 else 0
    print(" MyPlayer as P1 wins:", p1)
    print(" MyPlayer as P2 wins:", p2)
