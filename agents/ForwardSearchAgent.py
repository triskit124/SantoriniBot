import Util
import copy
import random
from .Player import Agent


class ForwardSearchAgent(Agent):
    """
    Implements a Forward Search agent to play Santorini. Forward search essentially enumerates every possible outcome to
    a certain depth (d_solve) and picks the most favorable action. As FS comes from the wold of Markov decision
    processes (MDPs), it does NOT consider any adversary actions. Thus, FS will win if left alone, but will lose easily
    if you are mean to it.

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

    def forwardSearch(self, board, position, d_solve, action_type):
        """
        Main implementation of the FS algorithm

        :param board: GameState representation of the current game board. See class GameState
        :param position: [3x1] list of [y,x,z] coordinates
        :param d_solve: solve depth
        :param action_type: string representing what type of turn it is, action_type = {'move' or 'build'}
        :return: a_star: greedy action corresponding to best value at root-node
        :return: v_star: value of the root node after forward search
        """

        # end states
        if d_solve == 0 or position[2] == 3:
            return None, 0

        a_star, v_star = None, -1e10
        if action_type == 'move':
            # move action space
            actions = Util.get_move_action_space(board, position)
            next_action = 'build'
        else:
            # build action space
            actions = Util.get_build_action_space(board, position)
            next_action = 'move'
        for action in actions:
            # take action and recurse
            v = self.reward(board, position, action)
            new_board, new_position = self.transition(board, position, action, self.player_number)
            a_prime, v_prime = self.forwardSearch(new_board, new_position, d_solve - 1, next_action)
            v = v + (self.gamma * v_prime)
            if v > v_star:
                a_star, v_star = action, v
        #print("d = ", d_solve, " position = ", position, " action= ", a_star, " v = ", v_star)
        return a_star, v_star

    def getAction(self, game, action_type):
        """
        Gets best action based on forward search. Essentially a wrapper function for forwardSearch()

        :param game: GameState representation of the current game board. See class GameState
        :return: action: greedy action corresponding to best value at root-node
        """
        board = copy.deepcopy(game.board)
        position = game.player_positions[self.player_number].copy()

        action, v = self.forwardSearch(board, position, self.d_solve, game.turn_type)
        return action

        self.gamma = config.getfloat('FS', 'gamma')                     # discount factor
        self.d_solve = config.getint('FS', 'd')                        # solve depth
        self.player_number = None                                       # marker for the board

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

    def transition(self, board, position, action, player_number):
        """
        Function to deterministically transition from current state to next state based on the action. Returns
        deep-copies of board and position.

        :param board: GameState representation of the current game board. See class GameState
        :param position: [3x1] list of [y,x,z] coordinates
        :param action: tuple of ('action', 'dir') where 'action' = {'move', 'build'} and 'dir' can be 'u', 'd', etc...
        :param player_number: int representing player index
        :return: new_board: deep-copied board for next state
        :return: new_position: copied position for next state
        """
        new_board = copy.deepcopy(board)

        if action[0] == 'move':
            # move
            old_position = position
            new_position = Util.move_logic(board, position, action)
            new_board[old_position[0]][old_position[1]][0] = None
            new_board[new_position[0]][new_position[1]][0] = player_number
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