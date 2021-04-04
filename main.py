import random


class Const:
    def __init__(self):
        self.gamma = 0.99


class GameState:
    def __init__(self):
        self.flag = 'un-initialized'
        self._board_size = 5
        self.board = [[['', 0] for i in range(self._board_size)] for j in range(self._board_size)]
        self.current_player_position = [0, 0, 0]
        self.current_opponent_position = [0, 0, 0]

    def print_board(self):
        for i in range(self._board_size):
            for j in range(self._board_size):
                print('(', self.board[i][j][0], ' , ', self.board[i][j][1], ')', '    ', end='')
            print('\n')

    def start_game(self):
        self.flag = 'started'
        self.current_player_position[0] = int(input("Welcome! Please place 1st Builder's row: "))
        self.current_player_position[1] = int(input("Please place 1st Builder's col: "))
        self.board[self.current_player_position[0]][self.current_player_position[1]][0] = 'B'

        self.print_board()


class Player:
    def __init__(self, position):
        self.position = position

    def check_move_validity(self, board, start_pos, end_pos):
        if board[end_pos[0]][end_pos[1]][0] != 'B' and board[end_pos[0]][end_pos[1]][0] != 'O' \
                and int(board[end_pos[0]][end_pos[1]][1]) <= int(start_pos[2]) + 1 \
                and end_pos[0] >= 0 and end_pos[1] >= 0 and board[end_pos[0]][end_pos[1]][1] < 4:
            return True
        return False

    def check_build_validity(self, board, build_pos):
        if build_pos[0] >= 0 and build_pos[1] >= 0 and board[build_pos[0]][build_pos[1]][1] < 4 \
                and board[build_pos[0]][build_pos[1]][0] != 'B' and board[build_pos[0]][build_pos[1]][0] != 'O':
            return True
        return False

    def move(self, board):
        movement = input("Select a move: Up (u), Down (d), Left (l), Right (r), Up-Left (ul), Up-Right (ur), Down-Left (dl), Down-Right (dr):   ")
        new_pos = self.position
        if movement == 'u':
            new_pos[0] -= 1
        elif movement == 'd':
            new_pos[0] += 1
        elif movement == 'r':
            new_pos[1] += 1
        elif movement == 'l':
            new_pos[1] -= 1
        elif movement == 'ur':
            new_pos[0] -= 1
            new_pos[1] += 1
        elif movement == 'dr':
            new_pos[0] += 1
            new_pos[1] += 1
        elif movement == 'ul':
            new_pos[0] -= 1
            new_pos[1] -= 1
        elif movement == 'dl':
            new_pos[0] += 1
            new_pos[1] -= 1

        new_pos[2] = board[new_pos[0]][new_pos[1]][1]

        if self.check_move_validity(board, self.position, new_pos):
            self.position = new_pos
        else:
            raise Exception("Not a valid move!")

    def build(self, board):
        building = input("Select where to build: Up (u), Down (d), Left (l), Right (r), Up-Left (ul), Up-Right (ur), Down-Left (dl), Down-Right (dr):   ")
        new_pos = self.position[0:2]
        print(new_pos)
        if building == 'u':
            new_pos[0] -= 1
        elif building == 'd':
            new_pos[0] += 1
        elif building == 'r':
            new_pos[1] += 1
        elif building == 'l':
            new_pos[1] -= 1
        elif building == 'ur':
            new_pos[0] -= 1
            new_pos[1] += 1
        elif building == 'dr':
            new_pos[0] += 1
            new_pos[1] += 1
        elif building == 'ul':
            new_pos[0] -= 1
            new_pos[1] -= 1
        elif building == 'dl':
            new_pos[0] += 1
            new_pos[1] -= 1

        if self.check_build_validity(board, new_pos):
            return new_pos
        else:
            raise Exception("Not a valid build!")


class Opponent:
    def __init__(self, game: GameState, policy_type):
        self.policy_type = policy_type
        self.game = game

    def generate_policy(self, board):
        policy = dict()
        if self.policy_type == 'random':
            avail = [[row, col] for row in range(self.game._board_size) for col in range(self.game._board_size) if board[row][col] == '.']
            policy = random.choices(avail)[0]

        return policy

    def move(self, board, game: GameState):
        current_policy = self.generate_policy(self, board)
        game.board[0] = current_move[0]
        game.board[1] = current_move[1]
        game.current_opponent_position = current_move


def player_move(game, player):
    old_player_position = player.position.copy()
    player.move(game.board)

    if player.position[2] == 3:
        game.flag = 'game_over'

    game.current_player_position = player.position
    game.board[old_player_position[0]][old_player_position[1]][0] = ''
    game.board[player.position[0]][player.position[1]][0] = 'B'

    game.print_board()


def player_build(game, player):
    build_loc = player.build(game.board)
    game.board[build_loc[0]][build_loc[1]][1] = game.board[build_loc[0]][build_loc[1]][1] + 1
    game.print_board()


def main():
    #run the program
    const = Const()
    game = GameState()
    game.start_game()

    player = Player(game.current_player_position)
    #opponent = Opponent(game, 'random')

    while game.flag != 'end':
        player_move(game, player)
        if game.flag == 'game_over':
            print("You Win!")
            break
        player_build(game, player)


if __name__ == '__main__':
    main()
