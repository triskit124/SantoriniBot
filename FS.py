import Util
import copy
import random
from MDP import MDP


class FSAgent(MDP):
    """
    Implements a Forward Search agent to play Santorini. Forward search essentially enumerates every possible outcome to
    a certain depth (d_solve) and picks the most favorable action. As FS comes from the wold of Markov decision
    processes (MDPs), it does NOT consider any adversary actions. Thus, FS will win if left alone, but will lose easily
    if you are mean to it.

    Inherits from parent class, MDP.
    """

    def __init__(self, config, player_number):
        super().__init__(config)
        self.player_number = player_number

    def choose_starting_position(self, board):
        """
        Function to choose a starting position on the board. Is called once during Game.start_game()

        :param board: GameState representation of the current game board. See class GameState
        :return: starting_position: a [3x1] List of [x, y, z] coordinates representing starting position
        """
        avail = [[row, col] for row in range(len(board[0])) for col in range(len(board[:][0])) if board[row][col][0] is None]
        position = random.choice(avail)
        return [position[0], position[1], 0]

    def forward_search(self, board, position, d_solve, action_type):
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
            a_prime, v_prime = self.forward_search(new_board, new_position, d_solve - 1, next_action)
            v = v + (self.gamma * v_prime)
            if v > v_star:
                a_star, v_star = action, v
        #print("d = ", d_solve, " position = ", position, " action= ", a_star, " v = ", v_star)
        return a_star, v_star

    def getAction(self, game):
        """
        Gets best action based on forward search. Essentially a wrapper function for forward_search()

        :param game: GameState representation of the current game board. See class GameState
        :return: action: greedy action corresponding to best value at root-node
        """
        board = copy.deepcopy(game.board)
        position = game.player_positions[self.player_number].copy()

        action, v = self.forward_search(board, position, self.d_solve, game.turn_type)
        return action

