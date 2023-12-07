from game import Game, Move, Player
import random
import numpy as np
from copy import deepcopy


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
    prev_value = deepcopy(board[(from_pos[1], from_pos[0])])
    acceptable = take1(board, (from_pos[1], from_pos[0]), player_id)
    if acceptable:
        acceptable = slide1(board, from_pos, slide)
        if not acceptable:
            board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
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


class MyPlayer(Player):
    def __init__(
        self,
        player_index,
        w_num_lines=100,
        w_num_corners=10,
        w_center_control=5,
        w_edge_control=3,
    ) -> None:
        super().__init__()
        self.index = player_index
        self.weights = {
            "num_lines": w_num_lines,
            "num_corners": w_num_corners,
            "center_control": w_center_control,
            "edge_control": w_edge_control,
        }

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

    def possible_takes(self, board, player):
        # top edge
        top = [(0, j) for j in range(5) if board[0, j] == player or board[0, j] == -1]
        # bottom edge
        bottom = [
            (4, j) for j in range(5) if board[4, j] == player or board[4, j] == -1
        ]
        # left edge
        left = [
            (i, 0) for i in range(1, 4) if board[i, 0] == player or board[i, 0] == -1
        ]
        # right edge
        right = [
            (i, 4) for i in range(1, 4) if board[i, 4] == player or board[i, 4] == -1
        ]

        return top + bottom + left + right

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        current_board = game._board
        current_heuristic = self.heuristic(current_board, self.index)

        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])

        b = current_board
        b_heuristic = current_heuristic
        explored = []

        for from_position in self.possible_takes(current_board, self.index):
            for slide in Move:
                # move1 already implements deepcopy to save state if movement is not possible
                # another deepcopy is need be added to restore if move is not optimal but this seems to work?
                if move1(b, from_position, slide, self.index):
                    if b not in explored:
                        explored.append(b)
                        explored += give_all_equivalent(b)
                        heuristic = self.heuristic(b, self.index)
                        if heuristic > b_heuristic:
                            b_heuristic = heuristic
                            from_pos = from_position
                            move = slide

        return from_pos, move
