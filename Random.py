import Util
import random


class RandomAgent:
    """
    Implements a Random agent to play Santorini. Nothing fancy here; the agent just lists all valid moves and chooses
    one randomly.
    """

    def __init__(self , agent_type="opponent"):
        self.agent_type = agent_type

    def getAction(self, game):
        """
        Selects a random action from a list of all valid actions

        :param game: GameState representation of the current game board. See class GameState
        :return: a random action
        """
        if self.agent_type == "opponent":
            position = game.opponent_position
        else:
            position = game.player_position

        if game.turn_type == 'move':
            #move
            actions = Util.get_move_action_space(game.board, position)
        else:
            # build
            actions = Util.get_build_action_space(game.board, position)

        if actions:
            action = random.choice(actions)
        else:
            action = None
        return action
