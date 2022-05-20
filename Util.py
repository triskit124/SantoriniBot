"""
A collection of useful utilities that are utilized by other classes.
"""


def move_logic(board, position, action):
    """
    Handles transitioning a position to a new position based on an action.

    :param board: GameState representation of the current game board. See class GameState
    :param position: [3x1] list of [y,x,z] coordinates
    :param action: tuple of ('action', 'dir') where 'action' = {'move', 'build'} and 'dir' can be 'u', 'd', etc...
    :return: new_pos: deep-copied new position after action
    """
    new_pos = position.copy()
    if action[1] == 'u':
        new_pos[0] -= 1
    elif action[1] == 'd':
        new_pos[0] += 1
    elif action[1] == 'r':
        new_pos[1] += 1
    elif action[1] == 'l':
        new_pos[1] -= 1
    elif action[1] == 'ur':
        new_pos[0] -= 1
        new_pos[1] += 1
    elif action[1] == 'dr':
        new_pos[0] += 1
        new_pos[1] += 1
    elif action[1] == 'ul':
        new_pos[0] -= 1
        new_pos[1] -= 1
    elif action[1] == 'dl':
        new_pos[0] += 1
        new_pos[1] -= 1

    if 0 <= new_pos[0] < len(board[:][0]) and 0 <= new_pos[1] < len(board[0][:]):
        new_pos[2] = board[new_pos[0]][new_pos[1]][1]

    return new_pos


def check_move_validity(board, start_pos, end_pos):
    """
    Check validity of a move based on board configuration and current position.

    :param board: GameState representation of the current game board. See class GameState
    :param start_pos: [3x1] list of [y,x,z] starting coordinates
    :param end_pos: [3x1] list of [y,x,z] ending coordinates
    :return: boolean representing if move is valid or not
    """
    if end_pos[0] < 0 or end_pos[0] >= len(board[:][0]) or end_pos[1] < 0 or end_pos[1] >= len(board[0][:]):
        return False
    if board[end_pos[0]][end_pos[1]][0] is None \
            and int(board[end_pos[0]][end_pos[1]][1]) <= int(start_pos[2]) + 1 \
            and end_pos[0] >= 0 and end_pos[1] >= 0 and board[end_pos[0]][end_pos[1]][1] < 4:
        return True
    return False


def check_build_validity(board, build_pos):
    """
    Check validity of a build based on board configuration and current position.

    :param board: GameState representation of the current game board. See class GameState
    :param build_pos: [3x1] list of [y,x,z] coordinates of where build is desired
    :return: boolean representing if build is valid or not
    """
    if build_pos[0] < 0 or build_pos[0] >= len(board[:][0]) or build_pos[1] < 0 or build_pos[1] >= len(board[0][:]):
        return False
    if build_pos[0] >= 0 and build_pos[1] >= 0 and board[build_pos[0]][build_pos[1]][1] < 4 \
            and board[build_pos[0]][build_pos[1]][0] is None:
        return True
    return False


def get_move_action_space(board, position):
    """
    Enumerates list of all valid move actions from current state

    :param board: GameState representation of the current game board. See class GameState
    :param position: [3x1] list of [y,x,z] coordinates representing current position
    :return: action_list: list of valid actions, where 'action' = {'move', 'build'} and 'dir' can be 'u', 'd', etc...
    """
    move_actions = [('move', 'u'), ('move', 'd'), ('move', 'l'), ('move', 'r'), ('move', 'ul'), ('move', 'ur'), ('move', 'dl'), ('move', 'dr')]
    action_list = []

    for action in move_actions:
        new_pos = move_logic(board, position, action)
        if check_move_validity(board, position, new_pos):
            action_list.append(action)
    return action_list


def get_build_action_space(board, position):
    """
    Enumerates list of all valid build actions from current state

    :param board: GameState representation of the current game board. See class GameState
    :param position: [3x1] list of [y,x,z] coordinates representing current position
    :return: action_list: list of valid actions, where 'action' = {'move', 'build'} and 'dir' can be 'u', 'd', etc...
    """
    build_actions = [('build', 'u'), ('build', 'd'), ('build', 'l'), ('build', 'r'), ('build', 'ul'), ('build', 'ur'), ('build', 'dl'), ('build', 'dr')]
    action_list = []

    for action in build_actions:
        new_pos = move_logic(board, position, action)
        if check_build_validity(board, new_pos):
            action_list.append(action)
    return action_list


def get_action_space(board, position, action_type):
    if action_type == 'move':
        return get_move_action_space(board, position)
    else:
        return get_build_action_space(board, position)


def what_is_next_turn(player_positions, agent, action_type):
    if action_type == 'move':
        next_agent = agent
        next_action = 'build'
    else:
        next_agent = (agent + 1) % len(player_positions)
        next_action = 'move'
    return next_agent, next_action


def get_all_actions(action_type):
    """
    Returns a list of all actions disregarding building/movement rules

    :param action_type: 'move' or 'build'
    :return: list of valid actions, where 'action' = {'move', 'build'} and 'dir' can be 'u', 'd', etc...
    """
    if action_type == 'move':
        return [('move', 'u'), ('move', 'd'), ('move', 'l'), ('move', 'r'), ('move', 'ul'), ('move', 'ur'), ('move', 'dl'), ('move', 'dr')]
    else:
        return [('build', 'u'), ('build', 'd'), ('build', 'l'), ('build', 'r'), ('build', 'ul'), ('build', 'ur'), ('build', 'dl'), ('build', 'dr')]


def boardToStringBoard(board, agent, action_type):
    """
    Converts a typical board representation into a string for Dict hashing using in MCTS
    :param board: GameState representation of the current game board. See class GameState
    :return: board_string: flattened string representation of board
    """
    import numpy as np
    import copy

    new_board = copy.deepcopy(board)
    for row in range(len(board)):
        for col in range(len(board)):

            # assign players (+1 if current player, -1 if opponent, 0 if empty)
            if board[row][col][0] is None:
                new_board[row][col][0] = 0
            elif board[row][col][0] == agent:
                new_board[row][col][0] = 1
            else:
                new_board[row][col][0] = -1

    return str(np.array(new_board).flatten())


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


def transition(board, player_positions, action, player_number):
    """
    Function to deterministically transition from current state to next state based on the action. Returns
    deep-copies of board and position.

    :param board: GameState representation of the current game board. See class GameState
    :param player_positions: list of [x,y,z] coordinates of each player
    :param action: tuple of ('action', 'dir') where 'action' = {'move', 'build'} and 'dir' can be 'u', 'd', etc...
    :param player_number: int, representing player index
    :return: new_board: deep-copied board for next state
    :return: new_position: copied position for next state
    """

    import copy

    new_board = copy.deepcopy(board)
    new_positions = copy.deepcopy(player_positions)

    if action[0] == 'move':
        old_position = player_positions[player_number]
        new_position = move_logic(board, old_position, action)

        new_positions[player_number] = new_position
        new_board[old_position[0]][old_position[1]][0] = None
        new_board[new_position[0]][new_position[1]][0] = player_number
    else:
        build_loc = move_logic(board, player_positions[player_number], action)
        new_board[build_loc[0]][build_loc[1]][1] = board[build_loc[0]][build_loc[1]][1] + 1
    return new_board, new_positions


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


