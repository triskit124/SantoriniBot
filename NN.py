import os.path

import numpy as np
import math
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import copy
import Util


class SantoriniNN(nn.Module):

    def __init__(self):

        self.board_x = 5 # TODO: hardcoded
        self.board_y = 5
        self.num_channels = 512
        self.dropout = 0.3
        self.batch_size = 32
        self.epochs = 10
        self.lr = 0.001

        super(SantoriniNN, self).__init__()

        # TODO: implement my own
        self.conv1 = nn.Conv2d(2, self.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        self.bn3 = nn.BatchNorm2d(self.num_channels)
        self.bn4 = nn.BatchNorm2d(self.num_channels)

        self.fc1 = nn.Linear(self.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, len(Util.get_all_actions('move')))

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        # TODO: implement my own
        s = s.view(-1, 2, self.board_x, self.board_y)  # batch_size x 2 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))  # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))  # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.num_channels * (self.board_x - 4) * (self.board_y - 4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


class NNAgent:

    def __init__(self, player_number):
        self.player_number = player_number
        self.model = SantoriniNN()

        # parameters for MCTS
        self.pi = []    # policy vector
        self.N = 25     # Number of MCTS sims
        self.c = 1    # Exploration bonus weight
        self.Qsa = {}   # Dict to store Q(s,a) values
        self.Nsa = {}   # Dict to store number of times s,a was visited
        self.Ns = {}    # Dict to store number of times board state s was visited
        self.Ps = {}    # initial policy (neural net)
        self.temp = 1   # temperature for policy sampling
        self.EPS = 1e-6 # small number to prevent singularities

    def MCTS(self, board, player_positions, agent, action_type):
        """
        Performs N Monte Carlo Tree Search simulations starting from the current board

        :param board:
        :param player_positions:
        :param agent:
        :param action_type:
        :return: pi: stochastic policy vector
        """
        for _ in range(self.N):
            self.search(board, player_positions, agent, action_type)

        all_actions = Util.get_all_actions(action_type)
        s = Util.boardToStringBoard(board)
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

        s = Util.boardToStringBoard(board) # get string representation of board for hashing

        # handle end states
        if player_positions[agent][2] == 3:
            return (-1) ** (agent != self.player_number) * 1.0  # return value based on who has won

        if not Util.get_action_space(board, player_positions[agent], action_type):
            return (-1) ** (agent == self.player_number) * 1.0  # return value based on who has won

        if s not in self.Ps:
            policy, v = self.predict(board)

            # get rid of any invalid actions
            all_actions = Util.get_all_actions(action_type)
            self.Ps[s] = [0] * len(all_actions)
            for i, p in enumerate(policy.tolist()[0]):
                if all_actions[i] in Util.get_action_space(board, player_positions[agent], action_type):
                    self.Ps[s][i] = p

            self.Ps[s] /= np.sum(self.Ps[s]) # re-normalize
            self.Ns[s] = 0
            return (-1) ** (agent != self.player_number) * v.tolist()[0][0]

        all_actions = Util.get_all_actions(action_type)
        valid_actions = Util.get_action_space(board, player_positions[agent], action_type)
        next_agent, next_action = Util.what_is_next_turn(player_positions, agent, action_type)

        if not valid_actions:
            return 0

        curr_best = -math.inf
        best_action = random.choice(valid_actions)

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
        return v

    def train(self, trainExamples):
        """
        Function to train the agent's neural network using training datasets

        :param trainExamples:
        :param loss_fn:
        :param optimizer:
        :return:
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.lr)
        num_batches = int(len(trainExamples) / self.model.batch_size)

        for epoch in range(self.model.epochs):
            self.model.train()
            print("\nTraining epoch {}...\n".format(epoch))

            for _ in range(num_batches):
                # pick out random samples
                # TODO
                idx = np.random.randint(len(trainExamples), size=self.model.batch_size)
                boards, turns, turn_types, pis, vs = list(zip(*[trainExamples[i] for i in idx]))
                boards = Util.boardsToNNBoard(boards)
                expected_pis = torch.FloatTensor(np.array(pis))
                expected_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # Predict
                # TODO
                predicted_pis, predicted_vs = self.model(boards)
                loss_pi = -torch.sum(expected_pis * predicted_pis) / expected_pis.size()[0]
                loss_v = torch.sum((expected_vs - predicted_vs.view(-1)) ** 2) / expected_vs.size()[0]
                total_loss = loss_pi + loss_v
                print(total_loss)

                # Backpropagation
                # TODO
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        board_size = len(board)
        board = Util.boardsToNNBoard([board])
        board = board.view(1, 2, board_size, board_size)

        self.model.eval()
        with torch.no_grad():
            pi, v = self.model(board)
        return pi, v # TODO

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

        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['state_dict'])

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
        torch.save({'state_dict': self.model.state_dict()}, filepath)

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




