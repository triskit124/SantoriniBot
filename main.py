import random


class Const:
    def __init__(self):
        self.gamma = 0.99


class GameState:
    def __init__(self):
        self.flag = 'starting'
        self._board_size = 5
        self.board = [['.' for i in range(self._board_size)] for j in range(self._board_size)]
        self.current_player_position = [0, 0]
        self.current_opponent_position = [0, 0]

    def print_board(self):
        for i in range(self._board_size):
            for j in range(self._board_size):
                print(self.board[i][j], '    ', end='')
            print('\n')

    def start_game(self):
        self.current_player_position[0] = int(input("Welcome! Please place 1st Builder's row: "))
        self.current_player_position[1] = int(input("Please place 1st Builder's col: "))
        self.board[self.current_player_position[0]][self.current_player_position[1]] = 'B1'

        #start_row = input("Please place 2nd Builder's row: ")
        #start_col = input("Please place 2nd Builder's col: ")
        #self.board[int(start_row)][int(start_col)] = 'B2'

        #place opponent builders
        #avail = [[row, col] for row in range(self._board_size) for col in range(self._board_size) if self.board[row][col] == '.']
        #rand_pos = random.choices(avail)[0]
        #self.board[rand_pos[0]][rand_pos[1]] = 'O1'

        self.print_board()


class Player:
    def __init__(self, position):
        self.position = position

    def move(self, board, movement):
        new_pos = self.position
        if movement == 'u':
            new_pos[0] -= 1
        elif movement == 'd':
            new_pos[0] += 1
        elif movement == 'r':
            new_pos[1] += 1
        elif movement == 'l':
            new_pos[1] -= 1
        elif movement == 'ur':
            new_pos[0] -= 1
            new_pos[1] += 1
        elif movement == 'dr':
            new_pos[0] += 1
            new_pos[1] += 1
        elif movement == 'ul':
            new_pos[0] -= 1
            new_pos[1] -= 1
        elif movement == 'dl':
            new_pos[0] += 1
            new_pos[1] -= 1

        if board[new_pos[0]][new_pos[1]] != 'B1' and board[new_pos[0]][new_pos[1]] != 'O1':
            self.position = new_pos

class Opponent:
    def __init__(self, game: GameState, policy_type):
        self.policy_type = policy_type
        self.game = game

    def generate_policy(self, board):
        policy = dict()
        if self.policy_type == 'random':
            avail = [[row, col] for row in range(self.game._board_size) for col in range(self.game._board_size) if board[row][col] == '.']
            policy = random.choices(avail)[0]

        return policy

    def move(self, board, game: GameState):
        current_policy = self.generate_policy(self, board)
        game.board[0] = current_move[0]
        game.board[1] = current_move[1]
        game.current_opponent_position = current_move



def main():
    #run the program
    const = Const()
    game = GameState()
    game.start_game()

    player = Player(game.current_player_position)
    opponent = Opponent(game, 'random')


    #while game.flag != 'end'
    #    game.player_turn()
    #    game.opponent_turn()


if __name__ == '__main__':
    main()
