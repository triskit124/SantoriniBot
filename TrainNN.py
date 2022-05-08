from Game import Player
from Arena import self_play
import ConfigHandler

import copy
import random
import pickle
import os
from matplotlib import pyplot as plt


class Coach:
    """
    A coach to train a neural network agent to play Santorini. The agent is trained by executing repeated games of self-
    play against an opponent.
    """

    def __init__(self, config):
        self.config = config
        self.numIters = config.getint('Train', 'num_iters') # number of training iterations
        self.numEps = config.getint('Train', 'num_eps') # number of training episodes
        self.arenaEps = config.getint('Train', 'arena_eps')
        self.folder = config['Train']['folder'] # directory to store training examples and model checkpoints
        self.updateThreshold = config.getfloat('Train', 'update_threshold') # win threshold above which to update neural network for agent during pit
        self.player0 = Player(config, policy_type=config['Game']['agent_0'], player_number=0) # candidate
        self.player1 = Player(config, policy_type=config['Game']['agent_1'], player_number=1) # opponent
        self.save_plots = config.getboolean('Train', 'save_plots')

        # Load pre-existing materials if available
        # ----------------------------------------

        if not os.path.exists(self.folder):
            print("Training directory does not exist. Making directory {}".format(self.folder))
            os.mkdir(self.folder)

        checkpoint_file = os.path.join(self.folder, 'best.pth.tar')
        if os.path.exists(checkpoint_file):
            print('Using pre-existing checkpoint file...')
            self.player0.Agent.loadCheckpoint(folder=self.folder, file='best.pth.tar')
            self.player1.Agent.loadCheckpoint(folder=self.folder, file='best.pth.tar')
        else:
            print('Starting training from scratch...')

        examples_file = os.path.join(self.folder, 'all_train_examples.tar')
        if os.path.exists(examples_file):
            print('Using pre-existing train examples...')
            self.trainExamplesHistory = self.loadFromPickle(folder=self.folder, file='all_train_examples.tar')
        else:
            print('Starting train examples from scratch...')
            self.trainExamplesHistory = []

        losses_file = os.path.join(self.folder, 'all_losses.tar')
        if os.path.exists(losses_file):
            print('Using pre-existing losses file...')
            self.all_losses = self.loadFromPickle(self.folder, 'all_losses.tar')
        else:
            print('Starting new losses file...')
            self.all_losses = []

        iteration_file = os.path.join(self.folder, 'last_iteration.tar')
        if os.path.exists(iteration_file):
            self.iteration = self.loadFromPickle(self.folder, 'last_iteration.tar')
            print('Using pre-existing iteration {}...'.format(self.iteration))
        else:
            self.iteration = 0
            print('Starting at iteration 0...')

        validation_file = os.path.join(self.folder, 'validation_set.tar')
        if os.path.exists(validation_file):
            print('Using validation set...')
            self.validationExamples = self.loadFromPickle(self.folder, 'validation_set.tar')
        else:
            self.validationExamples = []

        validation_losses_file = os.path.join(self.folder, 'validation_losses.tar')
        if os.path.exists(validation_losses_file):
            print('Using pre-existing validation losses file...')
            self.validation_losses = self.loadFromPickle(self.folder, 'validation_losses.tar')
        else:
            print('Starting new validation losses file...')
            self.validation_losses = []

    def executeEpisode(self, num_episodes=1):
        """
        Executes a single game of self-play to conclusion. Two NNAgents are pitted against each other.

        :return: trainExamples: list of tuples of form (board, current_player, turn_type, pi (policy), victory)
        """
        # reset MCTS trees before the agents start playing a new round of games
        self.player0.Agent.initialize_MCTS()
        self.player1.Agent.initialize_MCTS()

        # play a game
        players = [self.player0, self.player1]
        return self_play(config, players, num_games=num_episodes)

    def train(self):
        """
        Performs many episodes of self-play to train NNAgents.
        :return:
        """

        # ITERATION
        for i in range(self.numIters):
            print("\nTraining Iteration {}\n".format(self.iteration))

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
            self.player0.Agent.train(copy.deepcopy(self.trainExamplesHistory), copy.deepcopy(self.validationExamples))

            # append losses from most recent training to losses file
            self.all_losses.extend(self.player0.Agent.losses)
            self.validation_losses.extend(self.player0.Agent.validation_losses)

            # plot losses to file
            if self.save_plots:
                fig = plt.figure()
                plt.xlabel('sample')
                plt.ylabel('loss')
                plt.plot(self.all_losses)
                fig.savefig(self.folder + '/iteration {} losses.jpeg'.format(self.iteration), transparent=False, dpi=300, bbox_inches="tight")

                fig = plt.figure()
                plt.xlabel('epoch')
                plt.ylabel('validation loss')
                plt.plot(self.validation_losses)
                fig.savefig(self.folder + '/iteration {} validation losses.jpeg'.format(self.iteration), transparent=False, dpi=300, bbox_inches="tight")

            # Pit the new player agent against old version
            summary = self.executeEpisode(num_episodes=self.arenaEps)  # execute an episode
            wins, losses = summary[0]["wins"], summary[0]["losses"]

            # Accept or reject new model based on win percentage
            win_fraction = (wins / (wins + losses))
            print("Win percentage is {}%".format(100 * win_fraction))
            if win_fraction > self.updateThreshold:
                # Accept the new model
                print("Accepting new model...")
                self.player0.Agent.saveCheckpoint(folder=self.folder, file="iteration_{}.pth.tar".format(self.iteration))
                self.player0.Agent.saveCheckpoint(folder=self.folder, file="best.pth.tar")
            else:
                print("Rejecting new model...")
                self.player0.Agent.loadCheckpoint(folder=self.folder, file="tmp.pth.tar")

            self.iteration += 1

        # save all training materials to file
        self.saveToPickle(self.trainExamplesHistory, folder=self.folder, file="all_train_examples.tar")
        self.saveToPickle(self.all_losses, folder=self.folder, file="all_losses.tar")
        self.saveToPickle(self.validation_losses, folder=self.folder, file="validation_losses.tar")
        self.saveToPickle(self.iteration, folder=self.folder, file="last_iteration.tar")
        ConfigHandler.save_config(os.path.join(self.folder, 'config.ini'), self.config)

    @staticmethod
    def saveToPickle(data, folder=".", file="examples.tar"):
        """
        Saves training examples to flat binary file.

        :param data:
        :param folder:
        :param file:
        :return:
        """
        filepath = os.path.join(folder, file)
        if not os.path.exists(folder):
            print("Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def loadFromPickle(self, folder=".", file="examples.tar"):
        """
        Loads training examples from flat binary file.

        :param folder:
        :param file:
        :return:
        """
        filepath = os.path.join(folder, file)
        if not os.path.exists(filepath):
            raise ("No file to load in path {}".format(filepath))  # complain if file doesn't exist

        with open(filepath, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    # load in config file
    config = ConfigHandler.read_config('config/train.ini')

    coach = Coach(config)
    coach.train()
