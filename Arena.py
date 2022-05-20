from Game import GameState, Player
import Util
import ConfigHandler
import copy


def self_play(config, players, num_games=1):

    summary = {"games": [], "trainExamples": []}
    for player_number, player in enumerate(players):
        summary[player_number] = {
            "policy_type": player.policy_type,
            "wins": 0,
            "losses": 0,
            "games_played": 0,
            "win_percentage": 0,
        }

    for i in range(num_games):

        #run the program
        print("Playing game {}...".format(i))
        game = GameState(config=config)
        game.start_game(players)

        game_train_examples = []
        turns = 0
        while game.flag != 'game_over':
            turns += 1
            for player in players:
                example = [copy.deepcopy(game.board), game.turn, game.turn_type, None, None]
                player.move(game)
                if player.policy_type == 'NN' or player.policy_type == 'MiniMax':
                    example[3] = player.Agent.pi
                game_train_examples.append(tuple(example))

                train_symmetries = Util.getTrainingSymmetries(example)
                game_train_examples.extend(train_symmetries)

                if game.flag == 'game_over':
                    break

                example = [copy.deepcopy(game.board), game.turn, game.turn_type, None, None]
                player.build(game)
                if player.policy_type == 'NN' or player.policy_type == 'MiniMax':
                    example[3] = player.Agent.pi
                game_train_examples.append(tuple(example))

                train_symmetries = Util.getTrainingSymmetries(example)
                game_train_examples.extend(train_symmetries)

                if game.flag == 'game_over':
                    break

        # write summary
        summary[game.winner]["wins"] += 1
        summary["games"].append((game.winner, turns))
        for loser in game.losers:
            summary[loser]["losses"] += 1
        for player in game.players:
            summary[player]["games_played"] += 1
            summary[player]["win_percentage"] = 100 * summary[player]["wins"] / summary[player]["games_played"]

        # return annotated training examples based on whom won
        summary["trainExamples"].extend([(t[0], t[1], t[2], t[3], (-1) ** (t[1] != game.winner)) for t in game_train_examples])  # assign win/loss

    return summary


if __name__ == '__main__':
    # load in config file
    config = ConfigHandler.read_config('config/simple.ini')

    players = [Player(config, policy_type=config['Game']['agent_{}'.format(i)], player_number=i) for i in range(config.getint('Game', 'num_players'))]
    summary = self_play(config, players, num_games=config.getint('Arena', 'num_games'))
    print(summary)

