import copy
import math
import Util


class MiniMaxAgent:
    """
    Implements a Mini-Max agent to play Santorini. Moves are given by calling self.getAction().
    Implements alpha-beta pruning. Leaf nodes are evaluated using a heuristic evaluation function in
    self.evaluation_function().
    """

    def __init__(self, agent_type="opponent"):
        self.d = 8                      # solve depth
        self.alpha = -math.inf          # max cutoff for min action
        self.beta = math.inf            # min cutoff for max action
        self.agent_type = agent_type

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
            old_opponent_position = position
            new_position = Util.move_logic(board, position, action)
            new_board[old_opponent_position[0]][old_opponent_position[1]][0] = ''
            new_board[new_position[0]][new_position[1]][0] = player_marker
        else:
            build_loc = Util.move_logic(board, position, action)
            new_board[build_loc[0]][build_loc[1]][1] = board[build_loc[0]][build_loc[1]][1] + 1
            new_position = copy.deepcopy(position)
        return new_board, new_position

    def evaluation_function(self, board, player_position, opponent_position):
        """
        Function to evaluate the value of a board based on heuristics ("expert" knowledge)

        :param board: GameState representation of the current game board. See class GameState
        :param player_position: [3x1] list of [y,x,z] coordinates for Player (human)
        :param opponent_position: [3x1] list of [y,x,z] coordinates for Opponent (AI)
        :return: heuristic_score: the value of the current board
        """
        heuristic_score = 0
        for x in range(len(board[0])):
            for y in range(len(board[0])):
                # add score if AI is closer to higher tiles
                ro = math.sqrt((opponent_position[0] - y) ** 2 + (opponent_position[1] - x) ** 2)
                if ro != 0:
                    heuristic_score += board[y][x][1] / ro

                # remove score if Human is closer to higher tiles
                rp = math.sqrt((player_position[0] - y) ** 2 + (player_position[1] - x) ** 2)
                if rp != 0:
                    heuristic_score -= board[y][x][1] / rp
        return heuristic_score

    def alphabeta(self, board, player_position, opponent_position, alpha, beta, d_solve, agent, action_type):
        """
        Implementation of mini-max search with alpha-beta pruning.

        :param board: GameState representation of the current game board. See class GameState
        :param player_position: [3x1] list of [y,x,z] coordinates for Player (human)
        :param opponent_position: [3x1] list of [y,x,z] coordinates for Opponent (AI)
        :param alpha: upper-bound cutoff for min ply
        :param beta:  lower-bound cutoff for max ply
        :param d_solve: solve depth
        :param agent: string representing who's turn it is, agent = {'player' or 'opponent'}
        :param action_type: string representing what type of turn it is, action_type = {'move' or 'build'}
        :return: value: value of the root node after minimax search
        :return: action: greedy action corresponding to best value at root-node
        """

        # end states
        if player_position[2] == 3:
            return -math.inf, None # if human is at level 3, this is a loss state for the AI
        if opponent_position[2] == 3:
            return math.inf, None # if AI is at level 3, this is a win state for the AI
        if d_solve == 0:
            return self.evaluation_function(board, player_position, opponent_position), None # return heuristic

        # minimizing agent
        if agent == 'player':
            value = math.inf
            values = []
            if action_type == 'move':
                actions = Util.get_move_action_space(board, player_position)
                next_agent = 'player'
                next_action = 'build'
            else:
                actions = Util.get_build_action_space(board, player_position)
                next_agent = 'opponent'
                next_action = 'move'
            if not actions:
                return self.evaluation_function(board, player_position, opponent_position), None
            for action in actions:
                new_board, new_player_position = self.transition(board, player_position, action, 'B')
                new_opponent_position = copy.deepcopy(opponent_position)
                value = min(value, self.alphabeta(new_board, new_player_position, new_opponent_position, alpha, beta, d_solve - 1, next_agent, next_action)[0])
                values.append(value)
                if value <= alpha:
                    break
                beta = min(beta, value)
            return value, actions[values.index(value)]

        # maximizing agent
        else:
            value = -math.inf
            values = []
            if action_type == 'move':
                actions = Util.get_move_action_space(board, opponent_position)
                next_agent = 'opponent'
                next_action = 'build'
            else:
                actions = Util.get_build_action_space(board, opponent_position)
                next_agent = 'player'
                next_action = 'move'
            if not actions:
                return self.evaluation_function(board, player_position, opponent_position), None
            for action in actions:
                new_board, new_opponent_position = self.transition(board, opponent_position, action, 'O')
                new_player_position = copy.deepcopy(player_position)
                value = max(value, self.alphabeta(new_board, new_player_position, new_opponent_position, alpha, beta, d_solve - 1, next_agent, next_action)[0])
                values.append(value)
                if value >= beta:
                    break
                alpha = max(alpha, value)
            return value, actions[values.index(value)]

    def getAction(self, game):
        """
        Gets best action based on minimax search with alpha-beta pruning. Essentially a wrapper function for alphabeta()

        :param game: GameState representation of the current game board. See class GameState
        :return: action: greedy action corresponding to best value at root-node
        """

        # TODO: this is hacky, but if this agent is in self-play and is actially a "Player", need to swap inputs into the alphabeta call
        # alphabeta() assumed the "opponent" is maximizer and "player" is minimizer for everything. Very confusing if you
        # want two minimaxes to play each other!
        if self.agent_type == "opponent":
            dummy_board = copy.deepcopy(game.board)
            dummy_opponent_position = copy.deepcopy(game.opponent_position)
            dummy_player_position = copy.deepcopy(game.player_position)
            turn = game.turn
            turn_type = game.turn_type
        else:
            dummy_board = copy.deepcopy(game.board)
            dummy_opponent_position = copy.deepcopy(game.player_position)
            dummy_player_position = copy.deepcopy(game.opponent_position)
            if game.turn == "player":
                turn = "opponent"
            else:
                turn = "player"
            turn_type = game.turn_type

        v, action = self.alphabeta(dummy_board, dummy_player_position, dummy_opponent_position, self.alpha, self.beta, self.d, turn, turn_type)
        #print(v)
        return action
