from Game import GameState, Player


def self_play(player_types, num_games=1):

    summary = {}
    summary["games"] = []
    for player_number, policy_type in enumerate(player_types):
        summary[player_number] = {
            "policy_type": policy_type,
            "wins": 0,
            "losses": 0,
            "games_played": 0,
            "win_percentage": 0,
        }

    for _ in range(num_games):

        #run the program
        players = []
        game = GameState()

        for player_number, policy_type in enumerate(player_types):
            players.append(Player(policy_type=policy_type, player_number=player_number))

        game.start_game(players)

        turns = 0
        while game.flag != 'game_over':
            turns += 1
            for player in players:
                player.move(game)

                if game.flag == 'game_over':
                    break

                player.build(game)

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

    return summary


if __name__ == '__main__':
    summary = self_play(["FS", "MiniMax"], num_games=10)
    print(summary)

