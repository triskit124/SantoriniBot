from .Player import Player, Agent
from Game import SantoriniGame, BoardLocation, ActionType, Action

class HumanAgent(Agent):
    """
    Implements a Human agent to play Santorini. Moves are given by calling self.getAction().
    Actions are simply given by asking the user to input a move/build via the keyboard
    """

    @staticmethod
    def getActionDirectionFromPosition(current_position: BoardLocation, new_position: BoardLocation):
        moves = []
        delta_x = new_position[0] - current_position[0]
        delta_y = new_position[1] - current_position[1]
        if delta_x < 0:
            moves.extend(["u"] * abs(delta_x))
        if delta_x > 0:
            moves.extend(["d"] * delta_x)
        if delta_y < 0:
            moves.extend(["l"] * abs(delta_y))
        if delta_y > 0:
            moves.extend(["r"] * delta_y)

        return "".join(moves)
    
    def getAction(self, game: SantoriniGame, player: Player, action_type: ActionType) -> Action:
        if action_type == "CHOOSE_STARTNG_POSITION":
            # ask for player to place their builder
            print(f"Welcome player {player.getPlayerNumber()}!")
            while True:
                starting_row = int(input("Please place Builder (row): "))
                starting_col = int(input("Please place Builder (col): "))
                action = Action(player, "CHOOSE_STARTNG_POSITION", (starting_row, starting_col, 0))
                if action in player.getAllValidStartingActions(game.board):
                    break
                print("Not a valid placement, try again\n")
            print("\n")
        elif action_type == "MOVE":
            while True:
                valid_actions = player.getAllValidMoveActions(game.board, game.player_positions[player])
                choices = {HumanAgent.getActionDirectionFromPosition(game.player_positions[player], action.location): action for action in valid_actions}
                choice = input(f"Select a move: {list(choices.keys())}:   ")
                print("\n")
                if choice in choices:
                    action = choices[choice]
                    break
                print("Not a valid move!")

        elif action_type == "BUILD":
            while True:
                valid_actions = player.getAllValidBuildActions(game.board, game.player_positions[player])
                choices = {HumanAgent.getActionDirectionFromPosition(game.player_positions[player], action.location): action for action in valid_actions}
                choice = input(f"Select a build: {list(choices.keys())}:   ")
                print("\n")
                if choice in choices:
                    action = choices[choice]
                    break
                print("Not a valid move!")
        else:
            raise NotImplementedError(action_type)
        return action

