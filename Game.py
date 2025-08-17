#! /usr/bin/env python3


from dataclasses import dataclass
from typing import Optional, Literal, TYPE_CHECKING


if TYPE_CHECKING:
    from agents.Player import Player


@dataclass
class BoardSpace:
    player: Optional["Player"] = None
    height: int = 0

# Type aliases
ActionType = Literal["CHOOSE_STARTNG_POSITION", "MOVE", "BUILD"]
Board = list[list[BoardSpace]]
BoardLocation = tuple[int, int, int]
PlayerPositions = dict["Player", BoardLocation]

@dataclass
class Action():
    player: "Player"
    player_location: BoardLocation
    action_type: ActionType
    action_direction: str
    action_location: BoardLocation


class SantoriniGame:
    """
    Implements the state of the game. 
    Keeps track of all state variables, 
    including player positions and the game board.
    """

    def __init__(self, players: list["Player"], board_size: int = 5, verbose: bool = True):

        # players
        self.players = players # ordered list of players
        self.player_positions: PlayerPositions = {player: (0, 0, 0) for player in self.players}
        self.winner: Optional["Player"] = None
        self.losers: set["Player"] = set()
        
        # board
        self.board_size = board_size    
        self.board: Board = [[BoardSpace(None, 0) for _ in range(self.board_size)] for _ in range(self.board_size)]

        self.verbose = verbose

    def printBoard(self):
        """
        Helper function to print the current game board to the command line.
        """

        # maps square height to emojis
        height_print_dict = {
            0: "\N{white large square}",
            1: "\U0001F7E8",
            2: "\U0001F7E6",
            3: "\U0001F7E5",
            4: "\U0001F535",
        }
        if self.verbose:
            print('\n')
            for row in range(self.board_size):
                for col in range(self.board_size):
                    space = self.board[row][col]
                    print(f'{space.player.getPlayerPiece() if space.player else "  "}{height_print_dict[space.height]}  ', end='')
                print('\n')

    def play(self):
        """
        Main game loop.
        """

        if self.verbose:
            print(r"""                                              
                   __       __        ___  __          __ 
                  / /  ___ / /____   / _ \/ /__ ___ __/ / 
                 / /__/ -_) __(_-<  / ___/ / _ `/ // /_/  
                /____/\__/\__/___/ /_/  /_/\_,_/\_, (_)   
                                               /___/     
            """)

        # choose starting positions
        for player in self.players:
            self.applyAction(player.getAction(self, "CHOOSE_STARTNG_POSITION"))

        # game loop
        while not self.isGameOver():
            for player in self.players:
                player.playTurn(self)
    
    def applyAction(self, action: Action):
        """
        Applies a given action to the game board.
        Checks for new winners or losers after the action gets applied.
        """
        self.board, self.player_positions = SantoriniGame.getBoardAfterAction(self.board, self.player_positions, action)
        
        if self.verbose:
            self.printBoard()

        self._checkForWinnersOrLosers()

    def isGameOver(self) -> bool:
        """
        Returns whether or not the game is over.
        """
        # I don't think there are ties in Santorini, so this seems like a sufficient check
        return self.winner is not None
    
    def _checkForWinnersOrLosers(self):
        """
        Checks for new winners or losers given the current state of the game.
        """
        for player, position in self.player_positions.items():
            if player.isWinner(self.board, position):
                self.winner = player
                self.losers = set(self.players) - set([self.winner])
                print(f"player {self.winner.getPlayerNumber()} wins!")
                return
            
            if player.isLoser(self.board, position):
                self.losers.add(player)
                self.board[position[0]][position[1]].player = None
                print(f"player {player.getPlayerNumber()} loses!")
            
            remaining_players = set(self.players) - self.losers
            if len(remaining_players) == 1:
                self.winner = remaining_players.pop()
                print(f"player {self.winner.getPlayerNumber()} wins!")

    @staticmethod
    def getBoardAfterAction(board: Board, player_positions: PlayerPositions, action: Action) -> tuple[Board, PlayerPositions]:
        """
        Simulates applying an action to a game board. 
        Does not actually apply changes to the active game board.
        Instead, returns copies of the resulting board and player positions.
        Useful for agents that need to roll out actions without affecting the active game state.
        """

        from copy import copy

        # shallow copy to avoid copying player objects
        # agents may have state that we don't want to duplicate
        new_positions = {player: copy(position) for player, position in player_positions.items()}
        new_board = []
        for i in range(len(board[:][0])):
            new_board.append([])
            for j in range(len(board[0][:])):
                new_board[i].append(BoardSpace(board[i][j].player, board[i][j].height))

        if action.action_type == "CHOOSE_STARTNG_POSITION" or action.action_type == "MOVE" :
            new_board[action.player_location[0]][action.player_location[1]].player = None
            new_board[action.action_location[0]][action.action_location[1]].player = action.player
            new_positions[action.player] = action.action_location
        elif action.action_type == "BUILD":
            new_board[action.action_location[0]][action.action_location[1]].height += 1
        else:
            raise NotImplementedError(action.action_type)
        return new_board, new_positions
