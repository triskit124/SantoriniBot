# SantoriniBot

## Creating some AIs to play the board game, [Santorini](https://boardgamegeek.com/boardgame/194655/santorini)

**_Note:_** I do not claim any affiliation with Santorini. All credit goes to the developers.

### Agents (from easiest to mediumest difficulty):
- **RandomAgent** (very easy)
  - See `Random.py` for implementation. 
  - Just chooses random actions!
- **FSAgent** (easy)
  - See `FS.py` for implementation. 
  - Implements Markov decision process (MDP) forward search.
  - Enumerates every possible state to a certain depth but doesn't consider your actions
  - Will win if left alone but is easy to beat if you're mean to it!
- **MiniMaxAgent** (medium)
  - See `MiniMax.py` for implementation. 
  - Implements mini-max search with alpha-beta pruning.
  - Enumerates every possible state to a certain depth and assumes you take optimal actions every turn!
  - Plays a decent game but plays very conservatively and can only look a few turns ahead.

### Launch the game: 
- Play against RandomAgent: `python3 Game.py --agent Random`
- Play against FSAgent: `python3 Game.py --agent FS`
- Play against MiniMaxAgent: `python3 Game.py --agent MiniMax`

### Player markers:
- Your player: B
- The opponent: O

### Controls:
For either moving or building:
- Up (u)
- Down (d)
- Left (l)
- Right (r)
- Up-Left (ul)
- Up-Right (ur)
- Down-Left (dl)
- Down-Right (dr)
