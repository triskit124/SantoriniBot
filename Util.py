"""
A collection of useful utilities that are utilized by other classes.
"""


def boardsToNNBoards(boards, turns, turn_types):
    """
    Converts a typical board representation into a representation for use in neural networks
    :param board: GameState representation of the current game board. See class GameState
    :return: board_string: flattened string representation of board
    """
    import torch
    num_examples = len(boards)
    new_boards = torch.zeros(num_examples, 3, len(boards[0]), len(boards[0]))

    for e in range(num_examples):
        board = boards[e]
        turn = turns[e]
        turn_type = turn_types[e]
        for row in range(len(board)):
            for col in range(len(board)):

                # assign turn type to layer 0 (-1 if move, +1 if build)
                if turn_type == 'move':
                    new_boards[e][0][row][col] = -1
                else:
                    new_boards[e][0][row][col] = 1

                # assign players to layer 1 (+1 if current player, -1 if opponent, 0 if empty)
                if board[row][col][0] is None:
                    new_boards[e][1][row][col] = 0
                elif board[row][col][0] == turn:
                    new_boards[e][1][row][col] = 1
                else:
                    new_boards[e][1][row][col] = -1

                # assign build height to layer 2
                new_boards[e][2][row][col] = board[row][col][1]

    return torch.Tensor(new_boards) # returns tensor of size [num_examples, 3, board_x, board_y]


def getTrainingSymmetries(example):
    """
    Creates symmetrical training examples by rotating/flipping board, since Santorini is invariant to rotations/mirrors.
    Will create 7 additional examples for each example

    :param example: tuple of form (board, current_player, turn_type, pi (policy), victory)
    :return: example_symmetries: list of tuples of form (board, current_player, turn_type, pi (policy), victory)
    """
    import copy
    import numpy as np

    example_symmetries = []
    board = copy.deepcopy(example[0])
    turn = example[1]
    turn_type = example[2]
    pi = copy.deepcopy(example[3])
    v = example[4]

    all_actions = get_all_actions(turn_type)
    action = all_actions[np.argmax(pi)]

    # map to rotate moves by 90 degrees CCW
    rotation_map = {
        'd': 'r',
        'r': 'u',
        'u': 'l',
        'l': 'd',
        'dr': 'ur',
        'ur': 'ul',
        'ul': 'dl',
        'dl': 'dr',
    }

    # map to flip moves up/down
    flip_ud_map = {
        'd': 'u',
        'r': 'r',
        'u': 'd',
        'l': 'l',
        'dr': 'ur',
        'ur': 'dr',
        'ul': 'dl',
        'dl': 'ul',
    }

    # map to flip moves left/right
    flip_lr_map = {
        'd': 'd',
        'r': 'l',
        'u': 'u',
        'l': 'r',
        'dr': 'dl',
        'ur': 'ul',
        'ul': 'ur',
        'dl': 'dr',
    }

    # rotate the board by 90 degrees CCW 3 times
    rotated_board = copy.deepcopy(board)
    rotated_action = copy.deepcopy(action)
    for i in range(3):
        rotated_board = np.rot90(rotated_board)
        rotated_action = (rotated_action[0], rotation_map[rotated_action[1]])
        rotated_pi = [1 if rotated_action == a else 0 for a in all_actions]
        example_symmetries.append((rotated_board.tolist(), turn, turn_type, rotated_pi, v))


    # flip u/d then l/r
    flipped_board = copy.deepcopy(board)
    flipped_action = copy.deepcopy(action)

    flipped_board = np.flipud(flipped_board)
    flipped_action = (flipped_action[0], flip_ud_map[flipped_action[1]])
    flipped_pi = [1 if flipped_action == a else 0 for a in all_actions]
    example_symmetries.append((flipped_board.tolist(), turn, turn_type, flipped_pi, v))

    flipped_board = np.fliplr(flipped_board)
    flipped_action = (flipped_action[0], flip_lr_map[flipped_action[1]])
    flipped_pi = [1 if flipped_action == a else 0 for a in all_actions]
    example_symmetries.append((flipped_board.tolist(), turn, turn_type, flipped_pi, v))

    # flip l/r then u/d
    flipped_board = copy.deepcopy(board)
    flipped_action = copy.deepcopy(action)

    flipped_board = np.fliplr(flipped_board)
    flipped_action = (flipped_action[0], flip_lr_map[flipped_action[1]])
    flipped_pi = [1 if flipped_action == a else 0 for a in all_actions]
    example_symmetries.append((flipped_board.tolist(), turn, turn_type, flipped_pi, v))

    flipped_board = np.flipud(flipped_board)
    flipped_action = (flipped_action[0], flip_ud_map[flipped_action[1]])
    flipped_pi = [1 if flipped_action == a else 0 for a in all_actions]
    example_symmetries.append((flipped_board.tolist(), turn, turn_type, flipped_pi, v))

    return example_symmetries


