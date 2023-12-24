from game import Game, Move
import random
import numpy as np


def get_random_possible_action(
    game: "Game", player_index: int
) -> tuple[tuple[int, int], Move]:
    """random move from all the possible ones in the form :  row,col , move"""
    board = game.get_board()
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
    board: list[list[int]], player_index: int
) -> list[tuple[tuple[int, int], Move]]:
    """return list of all the possible moves for the player in the form :  list[row,col , move]"""
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


def try_move(
    action: tuple[tuple[int, int], Move], board: list[list[int]], player: int
) -> list[list[int]]:
    """Try to move the piece"""

    # unpack the action
    from_pos, slide = action

    # move passed should be from get_all_possible_actions
    if (from_pos, slide) not in get_all_possible_actions(board, player):
        return None

    # invert from_pos cause wtf
    from_pos = (from_pos[1], from_pos[0])

    # take the piece -> change its value to the player's index
    board[from_pos] = player
    piece = board[from_pos]

    # slide the board

    if slide == Move.LEFT:
        for i in range(from_pos[1], 0, -1):
            board[(from_pos[0], i)] = board[(from_pos[0], i - 1)]
        board[(from_pos[0], 0)] = piece

    elif slide == Move.RIGHT:
        for i in range(from_pos[1], board.shape[1] - 1, 1):
            board[(from_pos[0], i)] = board[(from_pos[0], i + 1)]
        board[(from_pos[0], board.shape[1] - 1)] = piece

    elif slide == Move.TOP:
        for i in range(from_pos[0], 0, -1):
            board[(i, from_pos[1])] = board[(i - 1, from_pos[1])]
        board[(0, from_pos[1])] = piece

    elif slide == Move.BOTTOM:
        for i in range(from_pos[0], board.shape[0] - 1, 1):
            board[(i, from_pos[1])] = board[(i + 1, from_pos[1])]
        board[(board.shape[0] - 1, from_pos[1])] = piece

    return board


def evaluate_winner(board: list[list[int]]) -> int:
    """Check the winner. Returns the player ID of the winner if any, otherwise returns -1"""
    # for each row
    for x in range(board.shape[0]):
        # if a player has completed an entire row
        if board[x, 0] != -1 and all(board[x, :] == board[x, 0]):
            return board[x, 0]

    # for each column
    for y in range(board.shape[1]):
        # if a player has completed an entire column
        if board[0, y] != -1 and all(board[:, y] == board[0, y]):
            return board[0, y]

    # if a player has completed the principal diagonal
    if board[0, 0] != -1 and all(
        [board[x, x] for x in range(board.shape[0])] == board[0, 0]
    ):
        return board[0, 0]

    # if a player has completed the secondary diagonal
    if board[0, -1] != -1 and all(
        [board[x, -(x + 1)] for x in range(board.shape[0])] == board[0, -1]
    ):
        return board[0, -1]
    return -1


if __name__ == "__main__":
    game = Game()
    # print(get_random_possible_action(game, 0))
    # print(get_all_possible_actions(game, 0))
    # action = get_random_possible_action(game, 0)
    # game.print()
    # print(action)
    # boardcpy_after_trying_action = try_move(game, action[0], action[1], game, 0)
    # print(boardcpy_after_trying_action)
    # print(evaluate_board(game, 0))
