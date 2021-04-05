import random
import time
import Util


class MDP:
    def __init__(self):
        self.gamma = 0.99

    def reward(self, board, position, action):
        new_pos = Util.move_logic(board, position, action)
        r = 0
        if action[0] == 'move':
            if new_pos[2] > position[2]:
                r = 1
            else:
                r = 1
            if new_pos[2] == 3:
                r = 1000
        elif action[0] == 'build':
            if new_pos[2] == 1:
                r = 1
            elif new_pos[2] == 2:
                r = 5
            elif new_pos[3] == 3:
                r = 10
        return r

    def get_move_action_space(self, board, position):
        move_actions = [('move', 'u'), ('move', 'd'), ('move', 'l'), ('move', 'r'), ('move', 'ul'), ('move', 'ur'), ('move', 'dl'), ('move', 'dr')]
        action_list = []

        for action in move_actions:
            new_pos = Util.move_logic(board, position, action)
            if Util.check_move_validity(board, position, new_pos):
                action_list.append(action)
        return action_list

    def get_build_action_space(self, board, position):
        build_actions = [('build', 'u'), ('build', 'd'), ('build', 'l'), ('build', 'r'), ('build', 'ul'), ('build', 'ur'), ('build', 'dl'), ('build', 'dr')]
        action_list = []

        for action in build_actions:
            new_pos = Util.move_logic(board, position, action)
            if Util.check_build_validity(board, new_pos):
                action_list.append(action)
        return action_list

    def transition(self, board, position, action):
        return Util.move_logic(board, position, action)


class GameState:
    def __init__(self):
        self.flag = 'un-initialized'
        self.board_size = 5
        self.board = [[['', 0] for i in range(self.board_size)] for j in range(self.board_size)]
        self.current_player_position = [0, 0, 0]
        self.current_opponent_position = [0, 0, 0]

    def print_board(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                print('(', self.board[i][j][0], ',', self.board[i][j][1], ')', '    ', end='')
            print('\n')

    def start_game(self):
        self.flag = 'started'
        self.current_player_position[0] = int(input("Welcome! Please place Builder (row): "))
        self.current_player_position[1] = int(input("Please place Builder (col): "))
        self.board[self.current_player_position[0]][self.current_player_position[1]][0] = 'B'

        avail = [[row, col] for row in range(self.board_size) for col in range(self.board_size) if self.board[row][col][0] == '']
        bot_position = random.choices(avail)[0]
        self.current_opponent_position[0] = bot_position[0]
        self.current_opponent_position[1] = bot_position[1]
        self.board[self.current_opponent_position[0]][self.current_opponent_position[1]][0] = 'O'

        self.print_board()


class Player:
    def __init__(self, position):
        self.position = position

    def move(self, game: GameState):
        old_player_position = self.position.copy()
        movement = input("Select a move: Up (u), Down (d), Left (l), Right (r), Up-Left (ul), Up-Right (ur), Down-Left (dl), Down-Right (dr):   ")
        new_pos = Util.move_logic(game.board, self.position, ('move', movement))

        if Util.check_move_validity(game.board, self.position, new_pos):
            self.position = new_pos
        else:
            raise Exception("Not a valid move!")

        if self.position[2] == 3:
            game.flag = 'game_over'

        game.current_player_position = self.position
        game.board[old_player_position[0]][old_player_position[1]][0] = ''
        game.board[self.position[0]][self.position[1]][0] = 'B'

        game.print_board()

    def build(self, game: GameState):
        building = input("Select where to build: Up (u), Down (d), Left (l), Right (r), Up-Left (ul), Up-Right (ur), Down-Left (dl), Down-Right (dr):   ")
        build_loc = Util.move_logic(game.board, self.position, ('build', building))

        if not Util.check_build_validity(game.board, build_loc):
            raise Exception("Not a valid build!")

        game.board[build_loc[0]][build_loc[1]][1] = game.board[build_loc[0]][build_loc[1]][1] + 1
        game.print_board()


class Opponent:
    def __init__(self, policy_type, position):
        self.policy_type = policy_type
        self.policy = dict()
        self.position = position

    def generate_policy(self, game: GameState):
        if self.policy_type == 'random':
            avail = [[row, col, game.board[row][col][1]] for row in range(game.board_size) for col in range(game.board_size) if game.board[row][col][0] == '' and abs(row - self.position[0]) == 1 and abs(col - self.position[1]) == 1]
            self.policy[str(game.board)] = random.choices(avail)[0]

    def move(self, game: GameState):
        old_opponent_position = self.position.copy()
        self.generate_policy(game)
        self.position = self.policy[str(game.board)]

        if self.position[2] == 3:
            game.flag = 'game_over'

        game.current_opponent_position = self.position
        game.board[old_opponent_position[0]][old_opponent_position[1]][0] = ''
        game.board[self.position[0]][self.position[1]][0] = 'O'

        print("Opponent is moving...")
        time.sleep(1)
        game.print_board()
        print('\n')

    def build(self, game: GameState):
        self.generate_policy(game)
        build_loc = self.policy[str(game.board)]
        game.board[build_loc[0]][build_loc[1]][1] = game.board[build_loc[0]][build_loc[1]][1] + 1

        print("Opponent is building...")
        time.sleep(1)
        game.print_board()
        print('\n')


def main():
    #run the program
    game = GameState()
    game.start_game()

    player = Player(game.current_player_position)
    opponent = Opponent('random', game.current_opponent_position)

    while game.flag != 'end':
        player.move(game)
        if game.flag == 'game_over':
            print("You Win!")
            break
        player.build(game)

        opponent.move(game)
        if game.flag == 'game_over':
            print("You Lose!")
            break
        opponent.build(game)


if __name__ == '__main__':
    main()
