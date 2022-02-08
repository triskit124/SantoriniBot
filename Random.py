import Util
import random


class RandomAgent:
    """
    Implements a Random agent to play Santorini. Nothing fancy here; the agent just lists all valid moves and chooses
    one randomly.
    """

    def __init__(self):
        pass

    def getAction(self, game):
        """
        Selects a random action from a list of all valid actions

        :param game: GameState representation of the current game board. See class GameState
        :return: a random action
        """
        if game.turn_type == 'move':
            #move
            actions = Util.get_move_action_space(game.board, game.opponent_position)
        else:
            # build
            actions = Util.get_build_action_space(game.board, game.opponent_position)

        if actions:
            action = random.choice(actions)
        else:
            action = None
        return action
