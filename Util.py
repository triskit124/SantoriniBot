def move_logic(board, position, action):
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
    if end_pos[0] < 0 or end_pos[0] >= len(board[:][0]) or end_pos[1] < 0 or end_pos[1] >= len(board[0][:]):
        return False
    if board[end_pos[0]][end_pos[1]][0] != 'B' and board[end_pos[0]][end_pos[1]][0] != 'O' \
            and int(board[end_pos[0]][end_pos[1]][1]) <= int(start_pos[2]) + 1 \
            and end_pos[0] >= 0 and end_pos[1] >= 0 and board[end_pos[0]][end_pos[1]][1] < 4:
        return True
    return False


def check_build_validity(board, build_pos):
    if build_pos[0] < 0 or build_pos[0] >= len(board[:][0]) or build_pos[1] < 0 or build_pos[1] >= len(board[0][:]):
        return False
    if build_pos[0] >= 0 and build_pos[1] >= 0 and board[build_pos[0]][build_pos[1]][1] < 4 \
            and board[build_pos[0]][build_pos[1]][0] != 'B' and board[build_pos[0]][build_pos[1]][0] != 'O':
        return True
    return False


def get_move_action_space(board, position):
    move_actions = [('move', 'u'), ('move', 'd'), ('move', 'l'), ('move', 'r'), ('move', 'ul'), ('move', 'ur'), ('move', 'dl'), ('move', 'dr')]
    action_list = []

    for action in move_actions:
        new_pos = move_logic(board, position, action)
        if check_move_validity(board, position, new_pos):
            action_list.append(action)
    return action_list


def get_build_action_space(board, position):
    build_actions = [('build', 'u'), ('build', 'd'), ('build', 'l'), ('build', 'r'), ('build', 'ul'), ('build', 'ur'), ('build', 'dl'), ('build', 'dr')]
    action_list = []

    for action in build_actions:
        new_pos = move_logic(board, position, action)
        if check_build_validity(board, new_pos):
            action_list.append(action)
    return action_list
