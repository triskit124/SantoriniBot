import copy
import Util


class HumanAgent:
    """
    Implements a Human agent to play Santorini. Moves are given by calling self.getAction().
    Actions are simply given by asking the user to input a move/build via the keyboard
    """

    def __init__(self, config, player_number):
        self.player_number = player_number

    def choose_starting_position(self, board):
        """
        Function to choose a starting position on the board. Is called once during Game.start_game()

        :param board: GameState representation of the current game board. See class GameState
        :return: starting_position: a [3x1] List of [x, y, z] coordinates representing starting position
        """
        # ask for player to place their builder
        print("Welcome player {}!".format(self.player_number))
        avail = [[row, col] for row in range(len(board[0])) for col in range(len(board[:][0])) if board[row][col][0] is None]
        while True:
            starting_row = int(input("Please place Builder (row): "))
            starting_col = int(input("Please place Builder (col): "))
            if [starting_row, starting_col] in avail:
                break
            print("Not a valid placement, try again\n")
        print("\n")
        return [starting_row, starting_col, 0]

    def get_move_action(self, game):
        """
        Execute move turn for Human player.

        :param game: GameState representation of the current game board. See class GameState
        """

        position = game.player_positions[self.player_number].copy()

        while True:
            movement = input("Select a move: Up (u), Down (d), Left (l), Right (r), Up-Left (ul), Up-Right (ur), Down-Left (dl), Down-Right (dr):   ")
            print("\n")
            action = ('move', movement)
            new_position = Util.move_logic(game.board, position, action)

            # check move validity
            if Util.check_move_validity(game.board, position, new_position):
                break
            print("Not a valid move!")

        return action

    def get_build_action(self, game):
        """
        Execute build turn for Human player.

        :param game: GameState representation of the current game board. See class GameState
        """

        position = game.player_positions[self.player_number].copy()

        while True:
            build = input("Select where to build: Up (u), Down (d), Left (l), Right (r), Up-Left (ul), Up-Right (ur), Down-Left (dl), Down-Right (dr):   ")
            print("\n")
            action = ('build', build)
            build_location = Util.move_logic(game.board, position, action)

            if Util.check_build_validity(game.board, build_location):
                break
            print("Not a valid build!")

        return action

    def getAction(self, game):
        if game.turn_type == 'move':
            action = self.get_move_action(game)
        else:
            action = self.get_build_action(game)
        return action

