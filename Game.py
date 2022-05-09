import argparse
import Util
import ConfigHandler

from FS import FSAgent
from Random import RandomAgent
from MiniMax import MiniMaxAgent
from HumanPlayer import HumanAgent
from NN import NNAgent


class GameState:
    """
    Implements the state of the game. Keeps track of all state variables, including player positions and the game board.
    """
    def __init__(self, config):

        self.config = config
        self.num_players = self.config.getint('Game', 'num_players')
        self.board_size = self.config.getint('Game', 'board_size')

        # players
        self.players = set(range(self.num_players))
        self.player_positions = [None for _ in range(self.num_players)]
        self.winner = None
        self.losers = set()

        # board
        self.board = [[[None, 0] for i in range(self.board_size)] for j in range(self.board_size)]

        # state
        self.flag = None # flag to keep track of game over state
        self.turn = None # [int] index that keeps track of whose turn it is
        self.turn_type = None # [string] what type of turn, 'move' or 'build'
        self.verbose = self.config.getboolean('Game', 'verbose') # print information to the console

    def print_board(self):
        """
        Helper function to print the current game board to the command line.
        """
        # maps player numbers to emojis
        player_print_dict = {
            None: "  ",
            0: "\U0001F477",
            1: "\U0001F916",
            2: "\U0001F916",
            3: "\U0001F916"
        }
        # maps square height to emojis
        height_print_dict = {
            0: "\N{white large square} ",
            1: "\U0001F7E8",
            2: "\U0001F7E6",
            3: "\U0001F7E5",
            4: "\U0001F535",
        }
        if self.verbose:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    print(player_print_dict[self.board[i][j][0]], height_print_dict[self.board[i][j][1]], '  ', end='')
                print('\n')

    def start_game(self, players):
        """
        Initializes the game.
        """
        if self.verbose:
            print("                                       \n \
                           __       __        ___  __          __ \n \
                          / /  ___ / /____   / _ \/ /__ ___ __/ / \n \
                         / /__/ -_) __(_-<  / ___/ / _ `/ // /_/  \n \
                        /____/\__/\__/___/ /_/  /_/\_,_/\_, (_)   \n \
                                                       /___/     ")
        self.turn_type = 'move' # start with move turn
        self.turn = 0 # start with player 0
        for player_number, player in enumerate(players):
            position = player.choose_starting_position(self.board) # choose starting position
            self.board[position[0]][position[1]][0] = player_number # update board
            self.player_positions[player_number] = position # update internally stored position

        self.print_board()

    def move_on_board(self, old_position, new_position, player_number):
        """
        Updates the game board to reflect a movement action.

        :param old_position: [3x1] list of [y,x,z] coordinates representing old position on board
        :param new_position: [3x1] list of [y,x,z] coordinates representing new position on board
        :param player_number: int associated with current player
        """
        self.board[old_position[0]][old_position[1]][0] = None
        self.board[new_position[0]][new_position[1]][0] = player_number
        self.player_positions[player_number] = new_position # update internally stored position
        self.print_board()
        self.turn_type = 'build' # next turn type is build after a move
        self.check_for_game_over()

    def build_on_board(self, build_position):
        """
        Updates the game board to reflect a build action.

        :param build_position: [3x1] list of [y,x,z] coordinates representing the desired build location
        """
        self.board[build_position[0]][build_position[1]][1] = self.board[build_position[0]][build_position[1]][1] + 1
        self.print_board()
        self.turn = (self.turn + 1) % self.num_players # switch to next player's turn after a build
        self.turn_type = 'move' # next turn type is move after a build
        self.check_for_game_over()

    def check_for_game_over(self):
        """
        Checks positions of each player and determines if a game_over state has been reached. This could be due to a
        player reaching a height of 3, or a player running out of valid moves.
        """

        for player_number, position in enumerate(self.player_positions):
            if player_number not in self.losers:
                # check if a player has reached a height of 3 (win condition)
                if position[2] == 3:
                    self.flag = 'game_over'
                    self.winner = player_number
                    self.losers = self.players - {self.winner}
                    print("Player {} wins!".format(player_number))
                    return
                # check if a player doesn't have any valid moves (that player loses)
                if not Util.get_move_action_space(self.board, position) or \
                   not Util.get_build_action_space(self.board, position):

                    self.losers.add(player_number)
                    self.board[position[0]][position[1]][0] = None
                    self.player_positions[player_number] = None

                    # if every player but 1 has lost, game is now over
                    if len(self.losers) == self.num_players - 1:
                        self.flag = 'game_over'
                        self.winner = (self.players - self.losers).pop() # use Set diff operation to find winner
                        print("Player {} wins!".format(self.winner))
                        return
                    else:
                        print("Player {} loses!".format(player_number))
        return


class Player:
    """
    Implements an Opponent to play Santorini against. Uses one of the various Agent implementations such as FS or
    MiniMax to generate moves.
    """
    def __init__(self, config, policy_type="Random", player_number=0):
        self.policy_type = policy_type
        self.player_number = player_number
        print(policy_type)

        # chose policy Agent
        if policy_type == 'FS':
            self.Agent = FSAgent(config, self.player_number)
        elif policy_type == 'Random':
            self.Agent = RandomAgent(config, self.player_number)
        elif policy_type == 'MiniMax':
            self.Agent = MiniMaxAgent(config, self.player_number)
        elif policy_type == 'Human':
            self.Agent = HumanAgent(config, self.player_number)
        elif policy_type == "NN":
            self.Agent = NNAgent(config, self.player_number)
        else:
            raise Exception("Invalid Agent selection '{}' for player {}!".format(policy_type, player_number))

    def choose_starting_position(self, board):
        """
        Function to choose a starting position on the board. Is called once during Game.start_game().
        This is a wrapper function for an Agent's specific choose_starting_positon() member function

        :param board: GameState representation of the current game board. See class GameState
        :return: starting_position: a [3x1] List of [x, y, z] coordinates representing starting position
        """
        return self.Agent.choose_starting_position(board)

    def move(self, game):
        """
        Execute a move turn for opponent. Chooses action generated by the policy Agent.
        :param game: GameState representation of the current game board. See class GameState
        """

        if game.verbose:
            print("\nPlayer {} is moving...\n".format(self.player_number))

        old_position = game.player_positions[self.player_number].copy()
        action = self.Agent.getAction(game) # get action from Agent
        new_position = Util.move_logic(game.board, old_position, action)
        game.move_on_board(old_position, new_position, self.player_number)

        if game.verbose:
            print('\n')

    def build(self, game):
        """
        Execute a build turn for opponent. Chooses action generated by the policy Agent.
        :param game: GameState representation of the current game board. See class GameState
        """

        if game.verbose:
            print("\nPlayer {} is building...\n".format(self.player_number))

        position = game.player_positions[self.player_number].copy()
        action = self.Agent.getAction(game)
        build_location = Util.move_logic(game.board, position, action)
        game.build_on_board(build_location)

        if game.verbose:
            print('\n')


def main():
    """
    Runs a game of Santorini with an AI adversary.
    """
    # parse command-line inputs (these will override config.ini settings)
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, help="Choose an opponent AI to play against", choices=['Random', 'FS', 'MiniMax'])
    parser.add_argument("--board_size", type=int, help="Board size", choices=[3, 4, 5])
    args = parser.parse_args()

    # load in config file
    config = ConfigHandler.read_config('config/simple.ini')

    # override config settings with command line args
    if args.board_size is not None:
        config['Game']['board_size'] = str(args.board_size)
    if args.agent is not None:
        config['Game']['agent_1'] = str(args.agent)

    #run the program
    #player = Player(policy_type="HumanAgent", player_number=0)
    #opponent = Player(policy_type=args.agent, player_number=1)
    players = [Player(policy_type=config['Game']['agent_{}'.format(i)], player_number=i) for i in range(config.getint('Game', 'num_players'))]

    game = GameState(config=config)
    game.start_game(players)

    # main game loop
    while game.flag != 'game_over':
        for player in players:
            if player.player_number not in game.losers:
                player.move(game)

                if game.flag == 'game_over':
                    break

                player.build(game)

                if game.flag == 'game_over':
                    break


if __name__ == '__main__':
    main()
