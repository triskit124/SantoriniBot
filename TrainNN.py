from Game import Player
from Arena import self_play

import copy
import random
import pickle
import os


class Coach:
    """
    A coach to train a neural network agent to play Santorini. The agent is trained by executing repeated games of self-
    play against an opponent.
    """

    def __init__(self, numIters=1, numEps=1, folder="training_session", starting_checkpoint=None, starting_train_examples=None):
        self.numIters = numIters # number of training iterations
        self.numEps = numEps # number of training episodes
        self.arenaEps = 10
        self.folder = folder # directory to store training examples and model checkpoints
        self.updateThreshold = 0.55 # win threshold above which to update neural network for agent during pit
        self.player0 = Player("NN", player_number=0) # candidate
        self.player1 = Player("NN", player_number=1) # opponent

        if starting_checkpoint is not None:
            self.player0.Agent.loadCheckpoint(file=starting_checkpoint)
            self.player1.Agent.loadCheckpoint(file=starting_checkpoint)

        if starting_train_examples is not None:
            self.trainExamplesHistory = self.loadTrainExamples(folder, starting_train_examples)
        else:
            self.trainExamplesHistory = []

    def executeEpisode(self, num_episodes=1):
        """
        Executes a single game of self-play to conclusion. Two NNAgents are pitted against each other.

        :return: trainExamples: list of tuples of form (board, current_player, turn_type, pi (policy), victory)
        """

        # play a game
        players = [self.player0, self.player1]
        return self_play(players, num_games=num_episodes)

    def train(self):
        """
        Performs many episodes of self-play to train NNAgents.
        :return:
        """

        # ITERATION
        for i in range(self.numIters):
            iterationTrainExamples = []
            print("\nTraining Iteration {}\n".format(i))

            # EPISODE
            summary = self.executeEpisode(num_episodes=self.numEps) # execute an episode
            iterationTrainExamples = summary["trainExamples"]

            # save iteration examples to history
            self.trainExamplesHistory.extend(iterationTrainExamples)

            # shuffle the training examples before learning
            random.shuffle(self.trainExamplesHistory)

            # store checkpoint and update opponent to old version
            self.player0.Agent.saveCheckpoint(folder=self.folder, file="tmp.pth.tar")
            self.player1.Agent.loadCheckpoint(folder=self.folder, file="tmp.pth.tar")

            # train the candidate agent
            self.player0.Agent.train(copy.deepcopy(self.trainExamplesHistory))

            # Pit the new player agent against old version
            summary = self.executeEpisode(num_episodes=self.arenaEps)  # execute an episode
            wins, losses = summary[0]["wins"], summary[0]["losses"]

            # Accept or reject new model based on win percentage
            if (wins / (wins + losses)) > self.updateThreshold:
                # Accept the new model
                print("Accepting new model...")
                self.player0.Agent.saveCheckpoint(folder=self.folder, file="iteration_{}.pth.tar".format(i))
                self.player0.Agent.saveCheckpoint(folder=self.folder, file="best.pth.tar")
            else:
                print("Rejecting new model...")
                self.player0.Agent.loadCheckpoint(folder=self.folder, file="tmp.pth.tar")

        # save all training examples to file
        self.saveTrainExamples(self.folder, "all_train_examples.tar")

    def saveTrainExamples(self, folder=".", file="examples.tar"):
        """
        Saves training examples to flat binary file.

        :param folder:
        :param file:
        :return:
        """
        filepath = os.path.join(folder, file)
        if not os.path.exists(folder):
            print("Train examples directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)

        with open(filepath, 'wb') as f:
            pickle.dump(self.trainExamplesHistory, f)

    def loadTrainExamples(self, folder=".", file="examples.tar"):
        """
        Loads training examples from flat binary file.

        :param folder:
        :param file:
        :return:
        """
        filepath = os.path.join(folder, file)
        if not os.path.exists(filepath):
            raise ("No train examples to load in path {}".format(filepath))  # complain if file doesn't exist

        with open(filepath, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    coach = Coach(numIters=10, numEps=5, folder="training_session", starting_checkpoint=None, starting_train_examples=None)
    coach.train()

