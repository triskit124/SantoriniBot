import random
import time

from Game import SantoriniGame, ActionType, Action
from .Player import Player, Agent


class RandomAgent(Agent):
    """
    Implements a Random agent to play Santorini. Nothing fancy here; the agent just lists all valid moves and chooses
    one randomly.
    """

    def getAction(self, game: SantoriniGame, player: Player, action_type: ActionType) -> Action:
        """
        Selects a random action from a list of all valid actions

        :param game: GameState representation of the current game board. See class GameState
        :return: a random action
        """

        time.sleep(1.0) # pause for a bit so we can see what's happening

        actions = player.getValidActions(game.board, game.player_positions[player], action_type)

        return random.choice(actions)

