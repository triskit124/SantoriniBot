from Player import Player
import random
import time


class RandomPlayer(Player):
    """
    Implements a Random agent to play Santorini. Nothing fancy here; the agent just lists all valid moves and chooses
    one randomly.
    """

    def chooseStartingPosition(self, board):
        """
        Function to choose a starting position on the board. Is called once during Game.start_game()

        :param board: GameState representation of the current game board. See class GameState
        :return: starting_position: a [3x1] List of [x, y, z] coordinates representing starting position
        """
        avail = [[row, col] for row in range(len(board[0])) for col in range(len(board[:][0])) if board[row][col][0] is None]
        position = random.choice(avail)
        return [position[0], position[1], 0]

    def getAction(self, game, action_type):
        """
        Selects a random action from a list of all valid actions

        :param game: GameState representation of the current game board. See class GameState
        :return: a random action
        """

        time.sleep(1.5) # pause for a bit so we can see what's happening

        actions = game.getAvailableActions()

        if actions:
            action = random.choice(actions)
        else:
            action = None
        return action
