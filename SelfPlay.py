import Game


def self_play(num_games=1, opponent_type="Random", player_type="MiniMax"):

    wins = 0
    losses = 0
    results = []
    summary = {}

    for _ in range(num_games):

        #run the program
        game = Game.GameState(game_type='self_play')
        game.start_game()

        opponent = Game.Opponent(game, opponent_type, self_play_type="opponent")
        player = Game.Opponent(game, player_type, self_play_type="player")

        turns = 0

        while True:
            turns += 1
            # AI player turn
            game.turn = 'player'
            game.turn_type = 'move'
            player.move(game)

            # check for player win
            if game.flag == 'game_won':
                print("You Win!")
                break
            elif game.flag == 'game_lost':
                print("you lost!")
                break

            game.turn_type = 'build'
            player.build(game)

            # AI opponent turn
            game.turn = 'opponent'
            game.turn_type = 'move'
            opponent.move(game)

            # check for opponent win
            if game.flag == 'game_lost':
                print("You Lose!")
                break
            elif game.flag == 'game_won':
                print("you won!")
                break

            game.turn_type = 'build'
            opponent.build(game)

        if game.flag == 'game_won':
            wins += 1
        elif game.flag == 'game_lost':
            losses += 1
        results.append((game.flag, turns))

    summary['wins'] = wins
    summary['losses'] = losses
    summary['games played'] = wins + losses
    summary['win percentage'] = 100 * wins / (wins + losses)
    return results, summary


if __name__ == '__main__':
    results, summary = self_play(num_games=20, opponent_type="FS", player_type="FS")

    print(results)
    print(summary)

