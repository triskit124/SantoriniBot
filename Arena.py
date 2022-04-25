from Game import GameState, Player
import copy


def self_play(players, num_games=1):

    summary = {}
    summary["games"] = []
    summary["trainExamples"] = []
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
        print("\nPlaying game {}...\n".format(i))
        game = GameState()
        game.start_game(players)

        game_train_examples = []
        turns = 0
        while game.flag != 'game_over':
            turns += 1
            for player in players:
                player.move(game)
                game_train_examples.append((copy.deepcopy(game.board), game.turn, game.turn_type, player.Agent.pi, None))

                if game.flag == 'game_over':
                    break

                player.build(game)
                game_train_examples.append((copy.deepcopy(game.board), game.turn, game.turn_type, player.Agent.pi, None))

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
    players = [
        Player("FS", player_number=0),
        Player("MiniMax", player_number=1)
    ]
    summary = self_play(players, num_games=10)
    print(summary)

