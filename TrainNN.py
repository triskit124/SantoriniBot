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
        self.num_validation_games = config.getint('Train', 'num_validation_games')
        self.folder = config['Train']['folder'] # directory to store training examples and model checkpoints
        self.updateThreshold = config.getfloat('Train', 'update_threshold') # win threshold above which to update neural network for agent during pit
        self.player0 = Player(config, policy_type=config['Game']['agent_0'], player_number=0) # candidate NN agent
        self.player1 = Player(config, policy_type=config['Game']['agent_1'], player_number=1) # opponent NN agent
        self.validation_player = Player(config, policy_type=config['Train']['validation_agent'], player_number=1) # validation opponent with known policy
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

        validation_win_rate_file = os.path.join(self.folder, 'validation_win_rates.tar')
        if os.path.exists(validation_win_rate_file):
            print('Using pre-existing validation win rate file...')
            self.validation_win_rates = self.loadFromPickle(self.folder, 'validation_win_rates.tar')
        else:
            print('Starting new validation win rate file...')
            self.validation_win_rates = []

    def executeEpisode(self, num_episodes=1, validation=False):
        """
        Executes a single game of self-play to conclusion. Two NNAgents are pitted against each other.

        :return: trainExamples: list of tuples of form (board, current_player, turn_type, pi (policy), victory)
        """
        # reset MCTS trees before the agents start playing a new round of games
        self.player0.Agent.initialize_MCTS()

        # play a game
        if validation:
            players = [self.player0, self.validation_player]
        else:
            self.player1.Agent.initialize_MCTS()
            players = [self.player0, self.player1]

        return self_play(config, players, num_games=num_episodes)

    def train(self):
        """
        Performs many episodes of self-play to train NNAgents.
        :return:
        """

        # ITERATION
        for i in range(self.numIters):
            print("\nTraining Iteration {}".format(self.iteration))

            # EPISODE
            print("\nPlaying games to produce training examples...")
            summary = self.executeEpisode(num_episodes=self.numEps, validation=True) #TODO execute episodes
            iterationTrainExamples = summary["trainExamples"]

            # save iteration examples to history
            self.trainExamplesHistory.extend(iterationTrainExamples)

            # shuffle the training examples before learning
            random.shuffle(self.trainExamplesHistory)

            # store checkpoint and update opponent to old version
            self.player0.Agent.saveCheckpoint(folder=self.folder, file="tmp.pth.tar")
            self.player1.Agent.loadCheckpoint(folder=self.folder, file="tmp.pth.tar")

            # train the candidate agent
            print("\nTraining with {} examples...".format(len(self.trainExamplesHistory)))
            self.player0.Agent.train(copy.deepcopy(self.trainExamplesHistory))

            # append losses from most recent training to losses file
            self.all_losses.extend(self.player0.Agent.losses)

            # plot losses to file
            if self.save_plots:
                fig = plt.figure()
                plt.xlabel('sample')
                plt.ylabel('loss')
                plt.plot(self.all_losses)
                fig.savefig(self.folder + '/iteration {} losses.jpeg'.format(self.iteration), transparent=False, dpi=300, bbox_inches="tight")

            # Pit the new player agent against old version
            print("\nPitting new model against previous...")
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

                # play games against validation opponent to test new model
                print("\nPlaying games against validation opponent...")
                summary = self.executeEpisode(num_episodes=self.num_validation_games, validation=True)
                wins, losses = summary[0]["wins"], summary[0]["losses"]
                win_fraction = (wins / (wins + losses))
                self.validation_win_rates.append(win_fraction)
                print("Win percentage against validation opponent is {}%".format(100 * win_fraction))

                # plot validation win rate to file
                if self.save_plots:
                    fig = plt.figure()
                    plt.xlabel('model version')
                    plt.ylabel('win rate')
                    plt.plot(self.validation_win_rates)
                    fig.savefig(self.folder + '/iteration_{}_validation_win_rates.jpeg'.format(self.iteration), transparent=False, dpi=300, bbox_inches="tight")

            else:
                print("Rejecting new model...")
                self.player0.Agent.loadCheckpoint(folder=self.folder, file="tmp.pth.tar")

            self.iteration += 1

        # save all training materials to file
        self.saveToPickle(self.trainExamplesHistory, folder=self.folder, file="all_train_examples.tar")
        self.saveToPickle(self.all_losses, folder=self.folder, file="all_losses.tar")
        self.saveToPickle(self.validation_win_rates, folder=self.folder, file="validation_win_rates.tar")
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
