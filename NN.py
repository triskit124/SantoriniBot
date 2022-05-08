import os.path
import numpy as np
import math
import random
import torch
from torch import nn
import torch.nn.functional as F

import copy
import Util


class SantoriniNN(nn.Module):
    """
    Implements a Neural Network used by the NNAgent class. Uses PyTorch as the representation and to do the heavy
    lifting.
    """

    def __init__(self, config):

        self.board_x = config.getint('Game', 'board_size')
        self.board_y = config.getint('Game', 'board_size')
        self.num_channels = config.getint('NN', 'num_channels')
        self.dropout = config.getfloat('NN', 'dropout')
        self.batch_size = config.getint('NN', 'batch_size')
        self.epochs = config.getint('NN', 'train_epochs')
        self.lr = config.getfloat('NN', 'lr')

        super(SantoriniNN, self).__init__()

        # TODO: took out layers to accommodate smaller board sizes
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1)
        #self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)
        #self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        #self.bn3 = nn.BatchNorm2d(self.num_channels)
        #self.bn4 = nn.BatchNorm2d(self.num_channels)

        #self.fc1 = nn.Linear(self.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024)
        self.fc1 = nn.Linear(self.num_channels * (self.board_x) * (self.board_y), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, len(Util.get_all_actions('move')))

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        # TODO: took out layers to accommodate smaller board sizes
        s = s.view(-1, 3, self.board_x, self.board_y)  # batch_size x 2 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))  # batch_size x num_channels x board_x x board_y
        #s = F.relu(self.bn3(self.conv3(s)))  # batch_size x num_channels x (board_x-2) x (board_y-2)
        #s = F.relu(self.bn4(self.conv4(s)))  # batch_size x num_channels x (board_x-4) x (board_y-4)
        #s = s.view(-1, self.num_channels * (self.board_x - 4) * (self.board_y - 4))
        s = s.view(-1, self.num_channels * (self.board_x) * (self.board_y))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


class NNAgent:
    """
    Implements a Neural Network agent to play Santorini. A CNN is used as a board evaluation function. Policies are
    created using Monte Carlo Tree Search (MCTS). The NN can be trained through self-play in TrainNN.py
    """

    def __init__(self, config, player_number):
        self.config = config
        self.player_number = player_number
        self.model = SantoriniNN(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.lr)
        self.epoch = 0
        self.losses = []
        self.validation_losses = []

        # set up parameters for MCTS
        self.initialize_MCTS()

    def initialize_MCTS(self):
        # parameters for MCTS
        self.pi = []                                        # policy vector
        self.N = self.config.getint('MCTS', 'N')            # Number of MCTS sims
        self.c = self.config.getfloat('MCTS', 'c')          # Exploration bonus weight
        self.Qsa = {}                                       # Dict to store Q(s,a) values
        self.Nsa = {}                                       # Dict to store number of times s,a was visited
        self.Ns = {}                                        # Dict to store number of times board state s was visited
        self.Ps = {}                                        # initial policy (neural net)
        self.temp = self.config.getfloat('MCTS', 'temp')    # temperature for policy sampling
        self.EPS = 1e-8                                     # small number to prevent singularities

    def MCTS(self, board, player_positions, agent, action_type):
        """
        Performs N Monte Carlo Tree Search simulations starting from the current board

        :param board:
        :param player_positions:
        :param agent:
        :param action_type:
        :return: pi: stochastic policy vector
        """
        for i in range(self.N):
            self.search(board, player_positions, agent, action_type)

        all_actions = Util.get_all_actions(action_type)
        s = Util.boardToStringBoard(board, agent, action_type)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in all_actions]
        bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
        bestA = np.random.choice(bestAs)
        pi = [0] * len(counts)
        pi[bestA] = 1
        self.pi = pi
        return pi

    def search(self, board, player_positions, agent, action_type):
        """
        Performs one iteration of MCTS.

        :param board:
        :param player_positions:
        :param agent
        :param action_type:
        :return:
        """

        s = Util.boardToStringBoard(board, agent, action_type) # get string representation of board for hashing
        next_agent, next_action = Util.what_is_next_turn(player_positions, agent, action_type)

        # handle end states
        if player_positions[agent][2] == 3:
            return 1 * (-1) ** (next_agent != agent) # return value based on who has won

        if not Util.get_action_space(board, player_positions[agent], action_type):
            return -1 * (-1) ** (next_agent != agent) # return value based on who has won

        if s not in self.Ps:
            policy, v = self.predict(board, agent, action_type)

            # get rid of any invalid actions
            all_actions = Util.get_all_actions(action_type)
            self.Ps[s] = [0] * len(all_actions)
            for i, p in enumerate(policy.tolist()[0]):
                if all_actions[i] in Util.get_action_space(board, player_positions[agent], action_type):
                    self.Ps[s][i] = p

            if np.sum(self.Ps[s]) == 0:
                print('Warning, all moved have been masked in MCTS')

            self.Ps[s] /= np.sum(self.Ps[s]) # re-normalize
            self.Ns[s] = 0
            return v.item() * (-1) ** (next_agent != agent)

        all_actions = Util.get_all_actions(action_type)
        valid_actions = Util.get_action_space(board, player_positions[agent], action_type)

        curr_best = -math.inf
        best_action = None

        # pick action to maximize UCB
        for i, a in enumerate(all_actions):
            if a in valid_actions:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.c * self.Ps[s][i] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.c * self.Ps[s][i] * math.sqrt(self.Ns[s] + self.EPS)

                if u > curr_best:
                    curr_best = u
                    best_action = a

        a = best_action
        new_board, new_positions = Util.transition(board, player_positions, a, agent)
        v = self.search(new_board, new_positions, next_agent, next_action)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v * (-1) ** (next_agent != agent)

    def train(self, trainExamples, validationExamples):
        """
        Function to train the agent's neural network using training datasets

        :param trainExamples:
        :param loss_fn:
        :param optimizer:
        :return:
        """
        num_batches = int(len(trainExamples) / self.model.batch_size)

        if validationExamples:
            #test loss with validation set
            val_boards, val_turns, val_turn_types, val_pis, val_vs = [], [], [], [], []
            for example in validationExamples:
                val_boards.append(example[0])
                val_turns.append(example[1])
                val_turn_types.append(example[2])
                val_pis.append(example[3])
                val_vs.append(example[4])

            val_boards = Util.boardsToNNBoards(val_boards, val_turns, val_turn_types)
            val_expected_pis = torch.FloatTensor(np.array(val_pis))
            val_expected_vs = torch.FloatTensor(np.array(val_vs).astype(np.float64))

        for epoch in range(self.model.epochs):
            self.model.train()
            print("Training epoch {}...".format(self.epoch))

            for batch in range(num_batches):
                # pick out random samples
                idx = np.random.randint(len(trainExamples), size=self.model.batch_size)
                boards, turns, turn_types, pis, vs = [], [], [], [], []
                for i in idx:
                    example = trainExamples[i]
                    boards.append(example[0])
                    turns.append(example[1])
                    turn_types.append(example[2])
                    pis.append(example[3])
                    vs.append(example[4])
                NN_boards = Util.boardsToNNBoards(boards, turns, turn_types)
                expected_pis = torch.FloatTensor(np.array(pis))
                expected_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # Predict
                # TODO
                predicted_pis, predicted_vs = self.model(NN_boards)
                loss_pi = -torch.sum(expected_pis * predicted_pis) / expected_pis.size()[0]
                loss_v = torch.sum((expected_vs - predicted_vs.view(-1)) ** 2) / expected_vs.size()[0]
                total_loss = loss_pi + loss_v
                self.losses.append(total_loss.item())
                print("Loss: {}".format(total_loss.item()))

                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            # Predict validation examples
            if validationExamples:
                val_predicted_pis, val_predicted_vs = self.model(val_boards)
                loss_pi = -torch.sum(val_expected_pis * val_predicted_pis) / val_expected_pis.size()[0]
                loss_v = torch.sum((val_expected_vs - val_predicted_vs.view(-1)) ** 2) / val_expected_vs.size()[0]
                total_validation_loss = loss_pi + loss_v
                self.validation_losses.append(total_validation_loss.item())
                print("Validation Loss: {}".format(total_validation_loss.item()))

            self.epoch += 1

    def predict(self, board, turn, turn_type):
        """
        Provides an estimate of a board's value and suggested policy using the Agent's NN model

        :param board:
        :return:
        """
        board_size = len(board)
        board = Util.boardsToNNBoards([board], [turn], [turn_type])
        board = board.view(1, 3, board_size, board_size)

        self.model.eval()
        with torch.no_grad():
            pi, v = self.model(board)
        return pi, v

    def loadCheckpoint(self, folder='.', file='tmp.pth.tar'):
        """
        Loads a PyTorch checkpoint.

        :param folder: directory to load file from
        :param file: filename in directory to load
        :return: None
        """
        filepath = os.path.join(folder, file)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath)) # complain if file doesn't exist

        print("Loading checkpoint from ", file)
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.losses = []
        self.validation_losses = []
        self.epoch = checkpoint['epoch']

    def saveCheckpoint(self, folder='.', file='tmp.pth.tar'):
        """
        Saves a PyTorch checkpoint.

        :param folder: directory to save file to
        :param file: filename in directory to save as
        :return: None
        """
        filepath = os.path.join(folder, file)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        print("Saving checkpoint to", file)
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
        }, filepath)

    def choose_starting_position(self, board):
        """
        Function to choose a starting position on the board. Is called once during Game.start_game()

        :param board: GameState representation of the current game board. See class GameState
        :return: starting_position: a [3x1] List of [x, y, z] coordinates representing starting position
        """
        avail = [[row, col] for row in range(len(board[0])) for col in range(len(board[:][0])) if board[row][col][0] is None]
        position = random.choice(avail)
        return [position[0], position[1], 0]

    def getAction(self, game):
        """
        Gets best action based on Monte Carlo Tree Search with neural network evaluation function

        :param game: GameState representation of the current game board. See class GameState
        :return: action: greedy action corresponding to best value at root-node
        """

        board_copy = copy.deepcopy(game.board)
        positions_copy = copy.deepcopy(game.player_positions)

        pi = self.MCTS(board_copy, positions_copy, game.turn, game.turn_type)
        all_actions = Util.get_all_actions(game.turn_type)
        action = all_actions[int(np.random.choice(len(all_actions), 1, p=pi))] # sample an action from policy
        return action




