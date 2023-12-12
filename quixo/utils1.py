from game import Move
import numpy as np
from copy import deepcopy

def test_move(original_board, move, player):
    """Test the move on the board and return the new board"""
    if player > 1:
        raise ValueError("Player must be 0 or 1")

    from_pos = move[0]
    slide = move[1]

    board = deepcopy(original_board)

    piece = board[from_pos]
    if slide == Move.TOP:
        for i in range(from_pos[1], 0, -1):
            board[(from_pos[0], i)] = board[(from_pos[0], 1 - 1)]
        board[(from_pos[0], 0)] = piece
    elif slide == Move.BOTTOM:
        for i in range(from_pos[1], board.shape[1], 1):
            board[(from_pos[0], i)] = board[(from_pos[0], 1 + 1)]
        board[(from_pos[0], board.shape[1] - 1)] = piece
    elif slide == Move.LEFT:
        for i in range(from_pos[0], 0, -1):
            board[(i, from_pos[1])] = board[(1 - 1, from_pos[1])]
        board[(0, from_pos[1])] = piece
    elif slide == Move.RIGHT:
        for i in range(from_pos[0], board.shape[0], 1):
            board[(i, from_pos[1])] = board[(1 + 1, from_pos[1])]
        board[(board.shape[0] - 1, from_pos[1])] = piece

    return board


def evaluate_board(board: list[list[int]], player: int) -> int:
    """Evaluate the board for the current player"""
    if player > 1:
        raise ValueError("Player must be 0 or 1")

    score_current = 0

    # weights
    corner_weight = 10
    edges_center_weight = 3
    core_weight = 5
    almost_complete_lines_weight = 100

    # owned cornes
    score_current += corner_weight if board[0][0] == player else 0
    score_current += corner_weight if board[0][4] == player else 0
    score_current += corner_weight if board[4][0] == player else 0
    score_current += corner_weight if board[4][4] == player else 0

    # edges cetner
    score_current += edges_center_weight if board[0][2] == player else 0
    score_current += edges_center_weight if board[2][0] == player else 0
    score_current += edges_center_weight if board[2][4] == player else 0
    score_current += edges_center_weight if board[4][2] == player else 0

    # core
    for i in range(1, 4):
        for j in range(1, 4):
            score_current += core_weight if board[i][j] == player else 0

    # almost complete lines
    for i in range(5):
        if np.count_nonzero(board[i] == player) == 4:
            score_current += almost_complete_lines_weight
        if np.count_nonzero(board[:, i] == player) == 4:
            score_current += almost_complete_lines_weight

    return score_current


def possible_takes(board, player):
    """return all possible take in the form of ((x,y),move)"""
    # top edge
    top = [(0, j) for j in range(5) if board[0, j] == player or board[0, j] == -1]
    top = [
        (coors, slide)
        for coors in top
        for slide in [Move.BOTTOM, Move.LEFT, Move.RIGHT]
    ]
    # bottom edge
    bottom = [(4, j) for j in range(5) if board[4, j] == player or board[4, j] == -1]
    bottom = [
        (coors, slide)
        for coors in bottom
        for slide in [Move.TOP, Move.LEFT, Move.RIGHT]
    ]
    # left edge
    left = [(i, 0) for i in range(1, 4) if board[i, 0] == player or board[i, 0] == -1]
    left = [
        (coors, slide)
        for coors in left
        for slide in [Move.TOP, Move.BOTTOM, Move.RIGHT]
    ]
    # right edge
    right = [(i, 4) for i in range(1, 4) if board[i, 4] == player or board[i, 4] == -1]
    right = [
        (coors, slide)
        for coors in right
        for slide in [Move.TOP, Move.BOTTOM, Move.LEFT]
    ]

    return top + bottom + left + right


def is_equivalent(board, target_board):
    """
    eight possible transformations that leave the game state unchanged:

    Identity (no change):

    The board remains the same.
    Horizontal Flip:

    Flip the board horizontally.
    Vertical Flip:

    Flip the board vertically.
    Diagonal Flip (from top-left to bottom-right):

    Swap the rows and columns.
    Diagonal Flip (from top-right to bottom-left):

    Reverse the order of the columns.
    Rotate 180 degrees:

    Combine a horizontal and vertical flip.
    Rotate 90 degrees clockwise:

    Rotate the board 90 degrees clockwise.
    Rotate 90 degrees counterclockwise:

    Rotate the board 90 degrees counterclockwise.
    """
    transformations = [
        lambda x: x,  # Identity
        lambda x: np.flipud(x),  # Horizontal Flip
        lambda x: np.fliplr(x),  # Vertical Flip
        lambda x: np.transpose(
            np.fliplr(x)
        ),  # Diagonal Flip (top-left to bottom-right)
        lambda x: np.flipud(np.fliplr(x)),  # Diagonal Flip (top-right to bottom-left)
        lambda x: np.rot90(x, k=2),  # Rotate 180 degrees
        lambda x: np.rot90(np.fliplr(x)),  # Rotate 90 degrees clockwise
        lambda x: np.rot90(np.flipud(x)),  # Rotate 90 degrees counterclockwise
    ]

    for transform in transformations:
        transformed_board = transform(board)
        if np.array_equal(transformed_board, target_board):
            return True
    return False


def give_all_equivalent(board):
    """return list of all equivalent boards"""
    transformations = [
        lambda x: x,  # Identity
        lambda x: np.flipud(x),  # Horizontal Flip
        lambda x: np.fliplr(x),  # Vertical Flip
        lambda x: np.transpose(
            np.fliplr(x)
        ),  # Diagonal Flip (top-left to bottom-right)
        lambda x: np.flipud(np.fliplr(x)),  # Diagonal Flip (top-right to bottom-left)
        lambda x: np.rot90(x, k=2),  # Rotate 180 degrees
        lambda x: np.rot90(np.fliplr(x)),  # Rotate 90 degrees clockwise
        lambda x: np.rot90(np.flipud(x)),  # Rotate 90 degrees counterclockwise
    ]

    equivalent_boards = []
    for transform in transformations:
        transformed_board = transform(board)
        equivalent_boards.append(transformed_board)
    return equivalent_boards


def move1(board, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
    """Perform a move"""
    if player_id > 2:
        return False
    # Oh God, Numpy arrays
    prev_value = board[(from_pos[1], from_pos[0])]
    acceptable = take1(board, (from_pos[1], from_pos[0]), player_id)
    if acceptable:
        acceptable = slide1(board, from_pos, slide)
        if not acceptable:
            board[(from_pos[1], from_pos[0])] = prev_value
    return acceptable


def take1(board, from_pos: tuple[int, int], player_id: int) -> bool:
    """Take piece"""
    # acceptable only if in border
    acceptable: bool = (
        (from_pos[0] == 0 and from_pos[1] < 5)
        or (from_pos[0] == 4 and from_pos[1] < 5)
        or (from_pos[1] == 0 and from_pos[0] < 5)
        or (from_pos[1] == 4 and from_pos[0] < 5)
        and (board[from_pos] < 0 or board[from_pos] == player_id)
    )
    if acceptable:
        board[from_pos] = player_id
    return acceptable


def slide1(board, from_pos: tuple[int, int], slide: Move) -> bool:
    """Slide the other pieces"""
    SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
    if from_pos not in SIDES:
        acceptable_top: bool = from_pos[0] == 0 and (
            slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
        )
        acceptable_bottom: bool = from_pos[0] == 4 and (
            slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
        )
        acceptable_left: bool = from_pos[1] == 0 and (
            slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
        )
        acceptable_right: bool = from_pos[1] == 4 and (
            slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
        )

    else:
        # top left
        acceptable_top: bool = from_pos == (0, 0) and (
            slide == Move.BOTTOM or slide == Move.RIGHT
        )
        # top right
        acceptable_right: bool = from_pos == (4, 0) and (
            slide == Move.BOTTOM or slide == Move.LEFT
        )
        # bottom left
        acceptable_left: bool = from_pos == (0, 4) and (
            slide == Move.TOP or slide == Move.RIGHT
        )
        # bottom right
        acceptable_bottom: bool = from_pos == (4, 4) and (
            slide == Move.TOP or slide == Move.LEFT
        )
    acceptable: bool = (
        acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
    )

    if acceptable:
        piece = board[from_pos]
        if slide == Move.TOP:
            for i in range(from_pos[1], 0, -1):
                board[(from_pos[0], i)] = board[(from_pos[0], 1 - 1)]
            board[(from_pos[0], 0)] = piece
        elif slide == Move.BOTTOM:
            for i in range(from_pos[1], board.shape[1], 1):
                board[(from_pos[0], i)] = board[(from_pos[0], 1 + 1)]
            board[(from_pos[0], board.shape[1] - 1)] = piece
        elif slide == Move.LEFT:
            for i in range(from_pos[0], 0, -1):
                board[(i, from_pos[1])] = board[(1 - 1, from_pos[1])]
            board[(0, from_pos[1])] = piece
        elif slide == Move.RIGHT:
            for i in range(from_pos[0], board.shape[0], 1):
                board[(i, from_pos[1])] = board[(1 + 1, from_pos[1])]
            board[(board.shape[0] - 1, from_pos[1])] = piece
    return acceptable
