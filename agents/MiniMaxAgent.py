import random
from math import inf
from typing import Optional

import numpy as np

from Game import SantoriniGame, Board, ActionType, Action, PlayerPositions
from .Player import Player, Agent


class MiniMaxAgent(Agent):
    """
    Implements a Mini-Max agent to play Santorini. Moves are given by calling self.getAction().
    Implements alpha-beta pruning.
    """

    def __init__(self, d_solve: int = 9):
        self._d_solve = d_solve

    def evaluationFunction(self, board: Board, player_positions: PlayerPositions, player: Player) -> float:
        """
        Function to evaluate the value of a board based on heuristics ("expert" knowledge)
        """
        player_idx = 1 if player.getAgent() is self else -1
        current_position = player_positions[player]

        return player_idx * current_position[2] * 10

    def alphaBeta(self, board: Board, player_positions: PlayerPositions, alpha: float, beta: float, d_solve: int, player: Player, action_type: ActionType) -> tuple[float, Optional[Action]]:
        """
        Implementation of mini-max search with alpha-beta pruning.

        """
        player_idx = 1 if player.getAgent() is self else -1

        # end states
        if player.isWinner(board, player_positions[player]):
            return player_idx * inf, None

        if player.isLoser(board, player_positions[player]):
            return player_idx * -inf, None

        if d_solve == 0:
            return self.evaluationFunction(board, player_positions, player), None # return heuristic
        
        # non-end state
        actions = player.getValidActions(board, player_positions[player], action_type)
        next_player = player.getNextPlayer(list(player_positions.keys()), action_type)
        next_action_type = player.getNextActionType(list(player_positions.keys()), action_type)

        value = player_idx * -inf
        values = []
        heuristic_values = [0.0] * len(actions)
        
        # order actions by heuristic value
        for i, action in enumerate(actions):
            new_board, new_positions = SantoriniGame.getBoardAfterAction(board, player_positions, action)
            # multiply by -player index because numpy argsort sorts from low to high. 
            # for minimizing player we want lowest scoring actions first
            # for maximizing player we want highest scoring actions first
            heuristic_values[i] = -player_idx * self.evaluationFunction(new_board, new_positions, player)

        ordered_actions = [actions[i] for i in np.argsort(heuristic_values)]

        for i, action in enumerate(ordered_actions):
            new_board, new_positions = SantoriniGame.getBoardAfterAction(board, player_positions, action)

            # minimizing agent
            if player.getAgent() is not self:
                new_value = self.alphaBeta(new_board, new_positions, alpha, beta, d_solve - 1, next_player, next_action_type)[0]
                values.append(new_value)
                value = min(value, new_value)
                if value <= alpha:
                    break
                beta = min(beta, value)
            
            # maximizing agent
            else:
                new_value = self.alphaBeta(new_board, new_positions, alpha, beta, d_solve - 1, next_player, next_action_type)[0]
                values.append(new_value)
                value = max(value, new_value)
                if value >= beta:
                    break
                alpha = max(alpha, value)
        
        selected_action = ordered_actions[values.index(value)]

        if d_solve == self._d_solve:
            print(f"{'Maximizing player' if player_idx == 1 else "Minimizing player"}, Solve depth {d_solve}")
            print(f"Selected action: {selected_action}, value: {value}, no. of pruned actions: {len(ordered_actions) - len(values)}, alpha: {alpha}, beta: {beta}")
            print("All actions:")
            for i, action in enumerate(ordered_actions):
                if i < len(values):
                    print(f"\tAction: {action}, value: {values[i]}")
                else:
                    print(f"\tAction: {action}, value: PRUNED")
        
        return value, selected_action


    def getAction(self, game: SantoriniGame, player: Player, action_type: ActionType) -> Action:
        """
        """
        if action_type == "CHOOSE_STARTNG_POSITION":
            return random.choice(player.getValidActions(game.board, game.player_positions[player], action_type))

        value, action = self.alphaBeta(game.board, game.player_positions, -inf, inf, self._d_solve, player, action_type)
        assert action
        return action
