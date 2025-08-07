#! /usr/bin/env python3

import os
import Util
from enum import Enum
from dataclasses import dataclass
from copy import copy
from typing import Iterable, Optional

from players.Player import Player
from FS import FSAgent
from Random import RandomAgent
from MiniMax import MiniMaxAgent
from HumanPlayer import HumanAgent
from NN import NNAgent


@dataclass
class SantoriniBoardSpace:
    player: Optional[Player] = None
    height: int = 0

class SantoriniActionType(Enum):
    CHOOSE_STARTNG_POSITION = 1
    MOVE = 2
    BUILD = 3

class SantoriniActionDirection(Enum):
    UP = "u"
    DOWN = "d"
    LEFT = "l"
    RIGHT = "r"
    UP_LEFT = "ul"
    DOWN_LEFT = "dl"
    UP_RIGHT = "ur"
    DOWN_RIGHT = "dr"

@dataclass
class SantoriniAction():
    action_type: SantoriniActionType
    direction: SantoriniActionDirection

# Type aliases
SantoriniBoard = list[list[SantoriniBoardSpace]]
SantoriniBoardLocation = tuple[int, int, int]

class SantoriniGame:
    """
    Implements the state of the game. 
    Keeps track of all state variables, 
    including player positions and the game board.
    """

    def __init__(self, players: list[Player], board_size: int = 5, verbose: bool = True):

        # players
        self.players = players # ordered list of players
        self.player_positions: dict[Player, SantoriniBoardLocation] = {player: (0, 0, 0) for player in self.players}
        self.winner: Optional[Player] = None
        self.losers: set[Player] = set()

        # board
        self.board_size = board_size    
        self.board: SantoriniBoard = [[SantoriniBoardSpace(None, 0) for _ in range(self.board_size)] for _ in range(self.board_size)]

        self.verbose = verbose

    def printBoard(self):
        """
        Helper function to print the current game board to the command line.
        """
        # maps player numbers to emojis
        player_print_dict = {
            None: "  ",
            0: "\U0001F477",
            1: "\U0001F916",
            2: "\U0001F916",
            3: "\U0001F916"
        }
        # maps square height to emojis
        height_print_dict = {
            0: "\N{white large square} ",
            1: "\U0001F7E8",
            2: "\U0001F7E6",
            3: "\U0001F7E5",
            4: "\U0001F535",
        }
        if self.verbose:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    space = self.board[i][j]
                    print(f'{space.player.getPlayerPiece() if space.player else "  "} {height_print_dict[space.height]}  ', end='')
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
            position = player.getAction(self, SantoriniActionType.CHOOSE_STARTNG_POSITION)
            self.board[position[0]][position[1]][0] = player # update board
            self.player_positions[player] = position # update internally stored position
        
        if self.verbose:
            self.printBoard()

        # game loop
        while True:
            for player in self.players:
                self.moveOnBoard(player)
                if self.checkForGameOver():
                    break
                self.buildOnBoard(player)
                if self.checkForGameOver():
                    break
    
    def moveOnBoard(self, player: Player):
        """
        Updates the game board to reflect a movement action.

        :param old_position: [3x1] list of [y,x,z] coordinates representing old position on board
        :param new_position: [3x1] list of [y,x,z] coordinates representing new position on board
        :param player_number: int associated with current player
        """
        if self.verbose:
            print(f"\nPlayer {player.getPlayerNumber()} is moving...\n")

        old_position = self.player_positions[player]
        new_position = player.getAction(self, SantoriniActionType.MOVE)

        self.board[old_position[0]][old_position[1]].player = None
        self.board[new_position[0]][new_position[1]].player = player
        self.player_positions[player] = new_position # update internally stored position

        if self.verbose:
            self.printBoard()

    def buildOnBoard(self, player: Player):
        """
        Updates the game board to reflect a build action.

        :param build_position: [3x1] list of [y,x,z] coordinates representing the desired build location
        """
        if self.verbose:
            print(f"\nPlayer {player.getPlayerNumber()} is building...\n")

        build_position = player.getAction(self, SantoriniActionType.BUILD)
        self.board[build_position[0]][build_position[1]].height += 1
        
        if self.verbose:
            self.printBoard()

    def checkForGameOver(self) -> bool:
        """
        Checks positions of each player and determines if a game_over state has been reached. This could be due to a
        player reaching a height of 3, or a player running out of valid moves.
        """

        for player in self.players:
            if player not in self.losers:
                # check if a player has reached a height of 3 (win condition)
                position = self.player_positions[player]
                if position[2] == 3:
                    self.winner = player
                    self.losers = set(self.players) - {self.winner}
                    print(f"Player {player.getPlayerNumber()} wins!")
                    return True
                # check if a player doesn't have any valid moves (that player loses)
                if not Util.get_move_action_space(self.board, position) or \
                   not Util.get_build_action_space(self.board, position):

                    self.players.remove(player)
                    self.player_positions.pop(player)
                    self.losers.add(player)
                    self.board[position[0]][position[1]].player = None

                    # if every player but 1 has lost, game is now over
                    if len(self.players) == 1:
                        self.winner = self.players[0]
                        print(f"Player {self.winner.getPlayerNumber()} wins!")
                        return True
                    else:
                        print(f"Player {player.getPlayerNumber()} loses!")
        return False
    
    @staticmethod
    def getPositionFromDirection(board: SantoriniBoard, pos: SantoriniBoardLocation, dir: SantoriniActionDirection) -> SantoriniBoardLocation:
        """
        Handles transitioning a position to a new position based on an action.

        :param board: GameState representation of the current game board. See class GameState
        :param position: [3x1] list of [y,x,z] coordinates
        :param action: tuple of ('action', 'dir') where 'action' = {'move', 'build'} and 'dir' can be 'u', 'd', etc...
        :return: new_pos: deep-copied new position after action
        """

        if dir == SantoriniActionDirection.UP:
            x = pos[0] - 1
            y = pos[1]
        elif dir == SantoriniActionDirection.DOWN:
            x = pos[0] + 1
            y = pos[1]
        elif dir == SantoriniActionDirection.RIGHT:
            x = pos[0]
            y = pos[1] + 1
        elif dir == SantoriniActionDirection.LEFT:
            x = pos[0]
            y = pos[1] - 1
        elif dir == SantoriniActionDirection.UP_RIGHT:
            x = pos[0] - 1
            y = pos[1] + 1
        elif dir == SantoriniActionDirection.DOWN_RIGHT:
            x = pos[0] + 1
            y = pos[1] + 1
        elif dir == SantoriniActionDirection.UP_LEFT:
            x = pos[0] - 1
            y = pos[1] - 1
        elif dir == SantoriniActionDirection.DOWN_LEFT:
            x = pos[0] + 1
            y = pos[1] - 1

        if not 0 <= x < len(board[:][0]) or not 0 <= y < len(board[0][:]):
            raise ValueError(f"Requested invalid board position: {x}, {y}")

        z = board[x][y].height

        return (x, y, z)

    @staticmethod
    def isValidMove(board: SantoriniBoard, start_pos: SantoriniBoardLocation, end_pos: SantoriniBoardLocation) -> bool:
        """

        """
        if end_pos[0] < 0 or end_pos[0] >= len(board[:][0]) or end_pos[1] < 0 or end_pos[1] >= len(board[0][:]):
            return False
        if board[end_pos[0]][end_pos[1]].player is None \
                and board[end_pos[0]][end_pos[1]].height <= start_pos[2] + 1 \
                and end_pos[0] >= 0 and end_pos[1] >= 0 and board[end_pos[0]][end_pos[1]].height <= 3:
            return True
        return False
    
    @staticmethod
    def isValidBoard(board: SantoriniBoard, build_pos: SantoriniBoardLocation) -> bool:
        """
        """
        if build_pos[0] < 0 or build_pos[0] >= len(board[:][0]) or build_pos[1] < 0 or build_pos[1] >= len(board[0][:]):
            return False
        if build_pos[0] >= 0 and build_pos[1] >= 0 and board[build_pos[0]][build_pos[1]].height <= 3 \
                and board[build_pos[0]][build_pos[1]].player is None:
            return True
        return False

    @staticmethod
    def getValidMoves(board: SantoriniBoard, position: SantoriniBoardLocation) -> list[SantoriniAction]:
        """
        Enumerates list of all valid move actions from current state

        :param board: GameState representation of the current game board. See class GameState
        :param position: [3x1] list of [y,x,z] coordinates representing current position
        :return: action_list: list of valid actions, where 'action' = {'move', 'build'} and 'dir' can be 'u', 'd', etc...
        """
        action_list = []

        for dir in SantoriniActionDirection: # TODO: does this work?
            new_pos = SantoriniGame.getPositionFromDirection(board, position, dir)
            if SantoriniGame.isValidMove(board, position, new_pos):
                action_list.append(action)
        return action_list