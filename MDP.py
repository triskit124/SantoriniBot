import copy
import math
import Util


class MDP:
    """
    Implementation of a base Markov decision process (MDP) class. Is inherited by specific MDP solver classes,
    such as FSAgent.
    """

    def __init__(self, gamma=0.99, d_solve=7):
        self.gamma = gamma          # discount factor
        self.d_solve = d_solve      # solve depth
        self.player_marker = 'O'    # marker for the board

    def reward(self, board, position, action):
        """
        Returns a reward based on the current state and action, R(s, a)

        :param board: GameState representation of the current game board. See class GameState
        :param position: [3x1] list of [y,x,z] coordinates
        :param action: tuple of ('action', 'dir') where 'action' = {'move', 'build'} and 'dir' can be 'u', 'd', etc...
        :return: reward: number
        """
        new_pos = Util.move_logic(board, position, action)
        r = 0
        if action[0] == 'move':
            if new_pos[2] > position[2]:
                r = 0
            else:
                r = 0
            if new_pos[2] == 3:
                r = 1000 # give a whole buncha reward for actually winning the game
        elif action[0] == 'build':
            new_pos[2] = board[new_pos[0]][new_pos[1]][1] + 1
            if new_pos[2] == 1:
                r = 0
            elif new_pos[2] == 2:
                r = 0
            elif new_pos[2] == 3:
                r = 0
        if r == 0:
            r = self.evaluation_function(board, position)
        return r

    def transition(self, board, position, action, player_marker):
        """
        Function to deterministically transition from current state to next state based on the action. Returns
        deep-copies of board and position.

        :param board: GameState representation of the current game board. See class GameState
        :param position: [3x1] list of [y,x,z] coordinates
        :param action: tuple of ('action', 'dir') where 'action' = {'move', 'build'} and 'dir' can be 'u', 'd', etc...
        :param player_marker: either 'O' for opponent (AI) or 'P' for player (human)
        :return: new_board: deep-copied board for next state
        :return: new_position: copied position for next state
        """
        new_board = copy.deepcopy(board)

        if action[0] == 'move':
            # move
            old_opponent_position = position
            new_position = Util.move_logic(board, position, action)
            new_board[old_opponent_position[0]][old_opponent_position[1]][0] = ''
            new_board[new_position[0]][new_position[1]][0] = player_marker
        else:
            # build
            build_loc = Util.move_logic(board, position, action)
            new_board[build_loc[0]][build_loc[1]][1] = board[build_loc[0]][build_loc[1]][1] + 1
            new_position = copy.deepcopy(position)
        return new_board, new_position

    def evaluation_function(self, board, position):
        """
        Function to evaluate the value of a board based on heuristics ("expert" knowledge)

        :param board: GameState representation of the current game board. See class GameState
        :param position: [3x1] list of [y,x,z] coordinates for Opponent (AI)
        :return: heuristic_score: the value of the current board
        """
        heuristic_score = 0
        for x in range(len(board[0])):
            for y in range(len(board[0])):
                r = math.sqrt((position[0] - y)**2 + (position[1] - x)**2)
                if r != 0:
                    heuristic_score += board[y][x][1] / r
        return heuristic_score
