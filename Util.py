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
    if board[end_pos[0]][end_pos[1]][0] != 'B' and board[end_pos[0]][end_pos[1]][0] != 'O' \
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
            and board[build_pos[0]][build_pos[1]][0] != 'B' and board[build_pos[0]][build_pos[1]][0] != 'O':
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

