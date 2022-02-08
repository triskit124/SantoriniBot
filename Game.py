import random
import copy
import argparse
import Util

from FS import FSAgent
from Random import RandomAgent
from MiniMax import MiniMaxAgent


class GameState:
    """
    Implements the state of the game. Keeps track of all state variables, including player positions and the game board.
    """
    def __init__(self, game_type='normal'):
        self.game_type = game_type
        self.flag = None
        self.board_size = 3
        self.board = [[['', 0] for i in range(self.board_size)] for j in range(self.board_size)]
        self.starting_player_position = [0, 0, 0]
        self.starting_opponent_position = [0, 0, 0]
        self.turn = None
        self.turn_type = None
        self.player_position = self.starting_player_position
        self.opponent_position = self.starting_opponent_position

        print("                                       \n \
               __       __        ___  __          __ \n \
              / /  ___ / /____   / _ \/ /__ ___ __/ / \n \
             / /__/ -_) __(_-<  / ___/ / _ `/ // /_/  \n \
            /____/\__/\__/___/ /_/  /_/\_,_/\_, (_)   \n \
                                           /___/     ")

    def print_board(self):
        """
        Helper function to print the current game board to the command line.
        """
        for i in range(self.board_size):
            for j in range(self.board_size):
                print(self.board[i][j][0], self.board[i][j][1], '   ', end='')
            print('\n')

    def start_game(self):
        """
        Initializes the game.
        """
        self.flag = 'started'

        # normal gameplay defines one Human player and one AI agent
        if self.game_type == 'normal':
            # ask for player to place their builder
            self.starting_player_position[0] = int(input("Welcome! Please place Builder (row): "))
            self.starting_player_position[1] = int(input("Please place Builder (col): "))
            print("\n")
            self.board[self.starting_player_position[0]][self.starting_player_position[1]][0] = 'B'

            # choose location to place opponent (just chooses randomly at this point in time)
            avail = [[row, col] for row in range(self.board_size) for col in range(self.board_size) if self.board[row][col][0] == '']
            bot_position = random.choice(avail)
            self.starting_opponent_position[0] = bot_position[0]
            self.starting_opponent_position[1] = bot_position[1]
            self.board[self.starting_opponent_position[0]][self.starting_opponent_position[1]][0] = 'O'

        # gameplay mode for playing two AI agents against one another
        elif self.game_type == 'self_play':
            avail = [[row, col] for row in range(self.board_size) for col in range(self.board_size) if self.board[row][col][0] == '']
            player_position = random.choice(avail)
            self.starting_player_position[0] = player_position[0]
            self.starting_player_position[1] = player_position[1]
            self.board[self.starting_player_position[0]][self.starting_player_position[1]][0] = 'B'

            avail = [[row, col] for row in range(self.board_size) for col in range(self.board_size) if self.board[row][col][0] == '']
            bot_position = random.choices(avail)[0]
            self.starting_opponent_position[0] = bot_position[0]
            self.starting_opponent_position[1] = bot_position[1]
            self.board[self.starting_opponent_position[0]][self.starting_opponent_position[1]][0] = 'O'

        self.print_board()

    def move_on_board(self, old_position, new_position, marker):
        """
        Updates the game board to reflect a movement action.

        :param old_position: [3x1] list of [y,x,z] coordinates representing old position on board
        :param new_position: [3x1] list of [y,x,z] coordinates representing new position on board
        :param marker: marker associated with current player, marker = {'B' or 'O'}
        """
        self.board[old_position[0]][old_position[1]][0] = ''
        self.board[new_position[0]][new_position[1]][0] = marker
        self.print_board()

    def build_on_board(self, build_position):
        """
        Updates the game board to reflect a build action.

        :param build_position: [3x1] list of [y,x,z] coordinates representing the desired build location
        """
        self.board[build_position[0]][build_position[1]][1] = self.board[build_position[0]][build_position[1]][1] + 1
        self.print_board()

    def update_player_position(self, position):
        """
        Updates internally stored position of the Human player

        :param position: [3x1] list of [y,x,z] coordinates representing position on board
        """
        self.player_position = position

    def update_opponent_position(self, position):
        """
        Updates internally stored position of the AI agent

        :param position: [3x1] list of [y,x,z] coordinates representing position on board
        """
        self.opponent_position = position

    def deepCopy(self):
        """
        Helper function to deep-copy the current GameState object. Not currently in use.
        :return: new_game: deep-copy of current GameState object
        """
        new_game = GameState()
        new_game.board = copy.deepcopy(self.board)
        new_game.flag = self.flag
        new_game.turn = self.turn
        new_game.turn_type = self.turn_type
        return new_game


class Player:
    """
    Implements a class to represent a Human player.
    """

    def __init__(self, game: GameState):
        self.position = game.starting_player_position
        self.player_marker = 'B'

    def move(self, game: GameState):
        """
        Execute move turn for Human player.

        :param game: GameState representation of the current game board. See class GameState
        """

        old_player_position = self.position.copy()

        while True:
            valid_moves = Util.get_move_action_space(copy.deepcopy(game.board), self.position)
            if not valid_moves:
                game.flag = "game_lost"
                break
            else:
                movement = input("Select a move: Up (u), Down (d), Left (l), Right (r), Up-Left (ul), Up-Right (ur), Down-Left (dl), Down-Right (dr):   ")
                print("\n")
                new_pos = Util.move_logic(game.board, self.position, ('move', movement))

                #check move validity
                if Util.check_move_validity(game.board, self.position, new_pos):
                    self.position = new_pos
                    break
                else:
                    print("Not a valid move!")

        if self.position[2] == 3:
            game.flag = 'game_won'

        game.move_on_board(old_player_position, self.position, self.player_marker)
        game.update_player_position(self.position)

    def build(self, game: GameState):
        """
        Execute build turn for Human player.

        :param game: GameState representation of the current game board. See class GameState
        """

        while True:
            building = input("Select where to build: Up (u), Down (d), Left (l), Right (r), Up-Left (ul), Up-Right (ur), Down-Left (dl), Down-Right (dr):   ")
            print("\n")
            build_loc = Util.move_logic(game.board, self.position, ('build', building))

            if Util.check_build_validity(game.board, build_loc):
                break
            print("Not a valid build!")

        game.build_on_board(build_loc)


class Opponent:
    """
    Implements an Opponent to play Santorini against. Uses one of the various Agent implementations such as FS or
    MiniMax to generate moves.
    """
    def __init__(self, game: GameState, policy_type, self_play_type="opponent"):
        self.policy_type = policy_type
        self.action = None
        self.self_play_type = self_play_type

        if self.self_play_type == "opponent":
            self.position = game.starting_opponent_position
            self.player_marker = 'O'
        elif self_play_type == "player":
            self.position = game.starting_player_position
            self.player_marker = 'B'
        else:
            self.position = None
            self.player_marker = None

        # chose policy Agent
        if policy_type == 'FS':
            self.Agent = FSAgent(agent_type=self.self_play_type)
        elif policy_type == 'Random':
            self.Agent = RandomAgent(agent_type=self.self_play_type)
        elif policy_type == 'MiniMax':
            self.Agent = MiniMaxAgent(agent_type=self.self_play_type)
        else:
            self.Agent = None

    def move(self, game: GameState):
        """
        Execute a move turn for opponent. Chooses action generated by the policy Agent.
        :param game: GameState representation of the current game board. See class GameState
        """

        if self.self_play_type == "opponent":
            print("\nOpponent is moving...\n")
        else:
            print("\nPlayer is moving...\n")

        old_opponent_position = self.position.copy()
        self.action = self.Agent.getAction(game) # get action from Agent

        if not self.action:
            if self.self_play_type == "opponent":
                game.flag = "game_won" # player has won
            else:
                game.flag = "game_lost" # player has lost
        else:
            self.position = Util.move_logic(game.board, self.position, self.action)
            if self.position[2] == 3:
                if self.self_play_type == "opponent":
                    game.flag = 'game_lost' # player has lost
                else:
                    game.flag = 'game_won' # player has won

            game.move_on_board(old_opponent_position, self.position, self.player_marker)

            if self.self_play_type == "opponent":
                game.update_opponent_position(self.position)
            else:
                game.update_player_position(self.position)

            print('\n')

    def build(self, game: GameState):
        """
        Execute a build turn for opponent. Chooses action generated by the policy Agent.
        :param game: GameState representation of the current game board. See class GameState
        """

        if self.self_play_type == "opponent":
            print("\nOpponent is building...\n")
        else:
            print("\nPlayer is building...\n")

        self.action = self.Agent.getAction(game)
        print(self.action)
        build_loc = Util.move_logic(game.board, self.position, self.action)
        game.build_on_board(build_loc)
        print('\n')


def main():
    """
    Runs a game of Santorini with an AI adversary.
    """

    # parse command-line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="Random", type=str, help="Choose an opponent AI to play against", choices=['Random', 'FS', 'MiniMax'])
    args = parser.parse_args()
    agent_select = args.agent

    #run the program
    game = GameState()
    game.start_game()

    player = Player(game)
    opponent = Opponent(game, agent_select)

    # main game loop
    while True:

        # Human player turn
        game.turn = 'player'
        game.turn_type = 'move'
        player.move(game)

        # check for player win/loss
        if game.flag == 'game_won':
            print("You Win!")
            break
        elif game.flag == 'game_lost':
            print("You Lose!")
            break

        game.turn_type = 'build'
        player.build(game)

        # AI opponent turn
        game.turn = 'opponent'
        game.turn_type = 'move'
        opponent.move(game)

        # check for opponent win
        if game.flag == 'game_lost':
            print("You Lose!")
            break
        elif game.flag == 'game_won':
            print("You Win!")
            break

        game.turn_type = 'build'
        opponent.build(game)


if __name__ == '__main__':
    main()
