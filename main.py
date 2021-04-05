import random
import time
import copy
import Util


class MDP:
    def __init__(self, gamma=0.99, d_solve=5):
        self.gamma = gamma
        self.d_solve = d_solve

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
            elif new_pos[2] == 3:
                r = 10
        return r

    def transition(self, board, position, action):
        new_board = copy.deepcopy(board)

        if action[0] == 'move':
            old_opponent_position = position
            new_position = Util.move_logic(board, position, action)
            new_board[old_opponent_position[0]][old_opponent_position[1]][0] = ''
            new_board[new_position[0]][new_position[1]][0] = 'O'
        else:
            build_loc = Util.move_logic(board, position, action)
            new_board[build_loc[0]][build_loc[1]][1] = board[build_loc[0]][build_loc[1]][1] + 1
            new_position = copy.deepcopy(position)
        return new_board, new_position

    def select_action(self, board, position, d_solve, action_type):
        if d_solve == 0:
            return 'NIL', 0
        a_star, v_star = 'NIL', -1e10
        if action_type == 'move':
            actions = Util.get_move_action_space(board, position)
            next_action = 'build'
        else:
            actions = Util.get_build_action_space(board, position)
            next_action = 'move'
        for action in actions:
            v = self.reward(board, position, action)
            new_board, new_position = self.transition(board, position, action)
            a_prime, v_prime = self.select_action(new_board, new_position, d_solve - 1, next_action)
            v = v + (self.gamma*v_prime)
            if v > v_star:
                a_star, v_star = action, v
        return a_star, v_star

    def forward_search(self, board, position, action_type):
        policy = dict()
        action, v = self.select_action(board, position, self.d_solve, action_type)
        #new_board, new_position = self.transition(board, position, action)
        policy[str(board)] = action
        return policy


class GameState:
    def __init__(self):
        self.flag = 'un-initialized'
        self.board_size = 5
        self.board = [[['', 0] for i in range(self.board_size)] for j in range(self.board_size)]
        self.starting_player_position = [0, 0, 0]
        self.starting_opponent_position = [0, 0, 0]

    def print_board(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                print(self.board[i][j][0], self.board[i][j][1], '   ', end='')
            print('\n')

    def start_game(self):
        self.flag = 'started'
        self.starting_player_position[0] = int(input("Welcome! Please place Builder (row): "))
        self.starting_player_position[1] = int(input("Please place Builder (col): "))
        self.board[self.starting_player_position[0]][self.starting_player_position[1]][0] = 'B'

        avail = [[row, col] for row in range(self.board_size) for col in range(self.board_size) if self.board[row][col][0] == '']
        bot_position = random.choices(avail)[0]
        self.starting_opponent_position[0] = bot_position[0]
        self.starting_opponent_position[1] = bot_position[1]
        self.board[self.starting_opponent_position[0]][self.starting_opponent_position[1]][0] = 'O'

        self.print_board()


class Player:
    def __init__(self, position):
        self.position = position

    def move(self, game: GameState):
        old_player_position = self.position.copy()

        while True:
            movement = input("Select a move: Up (u), Down (d), Left (l), Right (r), Up-Left (ul), Up-Right (ur), Down-Left (dl), Down-Right (dr):   ")
            new_pos = Util.move_logic(game.board, self.position, ('move', movement))

            if Util.check_move_validity(game.board, self.position, new_pos):
                self.position = new_pos
                break
            else:
                print("Not a valid move!")

        if self.position[2] == 3:
            game.flag = 'game_over'

        game.board[old_player_position[0]][old_player_position[1]][0] = ''
        game.board[self.position[0]][self.position[1]][0] = 'B'

        game.print_board()

    def build(self, game: GameState):
        while True:
            building = input("Select where to build: Up (u), Down (d), Left (l), Right (r), Up-Left (ul), Up-Right (ur), Down-Left (dl), Down-Right (dr):   ")
            build_loc = Util.move_logic(game.board, self.position, ('build', building))

            if Util.check_build_validity(game.board, build_loc):
                break
            print("Not a valid build!")

        game.board[build_loc[0]][build_loc[1]][1] = game.board[build_loc[0]][build_loc[1]][1] + 1
        game.print_board()


class Opponent:
    def __init__(self, policy_type, position):
        self.policy_type = policy_type
        self.policy = dict()
        self.position = position

    def generate_policy(self, game: GameState, action_type):
        if self.policy_type == 'random':
            if action_type == 'move':
                actions = Util.get_move_action_space(game.board, self.position)
            else:
                actions = Util.get_build_action_space(game.board, self.position)
            action = random.choices(actions)[0]
            self.policy[str(game.board)] = action
        elif self.policy_type == 'FS':
            mdp = MDP()
            dummy_board = copy.deepcopy(game.board)
            dummy_position = copy.deepcopy(self.position)
            self.policy = mdp.forward_search(dummy_board, dummy_position, action_type)

    def move(self, game: GameState):
        old_opponent_position = self.position.copy()
        self.generate_policy(game, 'move')
        self.position = Util.move_logic(game.board, self.position, self.policy[str(game.board)])

        if self.position[2] == 3:
            game.flag = 'game_over'

        game.board[old_opponent_position[0]][old_opponent_position[1]][0] = ''
        game.board[self.position[0]][self.position[1]][0] = 'O'

        print("Opponent is moving...")
        time.sleep(1)
        game.print_board()
        print('\n')

    def build(self, game: GameState):
        self.generate_policy(game, 'build')
        build_loc = Util.move_logic(game.board, self.position, self.policy[str(game.board)])
        game.board[build_loc[0]][build_loc[1]][1] = game.board[build_loc[0]][build_loc[1]][1] + 1

        print("Opponent is building...")
        time.sleep(1)
        game.print_board()
        print('\n')


def main():
    #run the program
    game = GameState()
    game.start_game()

    player = Player(game.starting_player_position)
    opponent = Opponent('FS', game.starting_opponent_position)

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
