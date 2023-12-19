from game import Game, Move
import random


def get_random_possible_action(
    game: "Game", player_index: int
) -> tuple[tuple[int, int], Move]:
    """random move from all the possible ones in the form :  row,col , move"""
    board = game._board
    row = None
    col = None
    move = None

    while True:
        is_row = random.choice([True, False])

        if is_row:
            row = random.randint(0, 4)
            col = random.choice([0, 4])
        else:
            col = random.randint(0, 4)
            row = random.choice([0, 4])

        if board[row, col] == -1 or board[row, col] == player_index:
            break

    possible_moves = [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]
    if row == 0:
        possible_moves.remove(Move.TOP)
    if row == 4:
        possible_moves.remove(Move.BOTTOM)
    if col == 0:
        possible_moves.remove(Move.LEFT)
    if col == 4:
        possible_moves.remove(Move.RIGHT)

    move = random.choice(possible_moves)

    return (row, col), move


def get_all_possible_actions(
    game: "Game", player_index: int
) -> list[tuple[tuple[int, int], Move]]:
    """return list of all the possible moves for the player in the form :  list[row,col , move]"""
    board = game._board
    actions = []

    # corners
    if board[0, 0] == -1 or board[0, 0] == player_index:
        actions.extend([((0, 0), Move.RIGHT), ((0, 0), Move.BOTTOM)])

    if board[4, 4] == -1 or board[4, 4] == player_index:
        actions.extend([((4, 4), Move.TOP), ((4, 4), Move.LEFT)])

    if board[4, 0] == -1 or board[4, 0] == player_index:
        actions.extend([((4, 0), Move.RIGHT), ((4, 0), Move.TOP)])

    if board[0, 4] == -1 or board[0, 4] == player_index:
        actions.extend([((0, 4), Move.LEFT), ((0, 4), Move.BOTTOM)])

    # edges
    for i in range(1, 4):
        if board[0, i] == -1 or board[0, i] == player_index:
            actions.extend(
                [((0, i), Move.LEFT), ((0, i), Move.RIGHT), ((0, i), Move.BOTTOM)]
            )

        if board[4, i] == -1 or board[4, i] == player_index:
            actions.extend(
                [((4, i), Move.LEFT), ((4, i), Move.RIGHT), ((4, i), Move.TOP)]
            )

        if board[i, 0] == -1 or board[i, 0] == player_index:
            actions.extend(
                [((i, 0), Move.TOP), ((i, 0), Move.RIGHT), ((i, 0), Move.BOTTOM)]
            )

        if board[i, 4] == -1 or board[i, 4] == player_index:
            actions.extend(
                [((i, 4), Move.TOP), ((i, 4), Move.LEFT), ((i, 4), Move.BOTTOM)]
            )

    return actions
