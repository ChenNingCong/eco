R-öko – A game of recycling and ecology

## Deck & State Space Notes (for RL implementation)

### Recycling Cards (88 total)
- 4 colors: Glass, Paper, Plastic, Tin
- 19 single cards (value = 1) per color  →  76 cards
- 3  double cards (value = 2) per color  →  12 cards
- Total: 88 recycling cards

### State Space
The state is represented as a perfect-information vector including:
- **Hands**: `(num_players, 4 colors, 2 types)` integer counts
- **Recycling side**: `(4 colors, 2 types)` counts of cards played this turn
- **Waste side**: `(4 factories, 4 colors, 2 types)` counts on each factory's waste pile
- **Factory stacks**: per-color list of remaining factory-card values (top-to-bottom)
- **Collected factory cards**: `(num_players, 4 colors)` sum of values taken per player
- **Penalty cards**: `(num_players,)` count of face-down excess cards
- **Draw pile size**: integer (exact composition tracked internally for reshuffling)

### Action Space (108 actions, phase-masked)
- **Play phase** (100 actions): `color × n_singles × n_doubles` where each ∈ {0..4}.
  Encoded as `color*25 + n_singles*5 + n_doubles`. The (0,0) entry per color is illegal.
- **Discard phase** (8 actions): `color × type` (discard one card), encoded as `100 + color*2 + type`.

### Turn Phase Token
A binary `phase` field (0 = play, 1 = discard) is included in the observation so the
agent knows which action subset is currently legal.

---

Game Idea
When night falls on the recycling center, carelessly discarded glass bottles, piles of paper, Plastic cups and tin cans
wake up and come to life and have a party with cockroaches, crickets and other pests. However, with the first light
of morning shift workers find only waste needing sorting again.
The players try to properly separate bottles, cans, cups and paper and bring them into the recycling factory to get a
lot of money. Unfortunately players often produce too much waste and then sometimes dispose of it illegally. The
winner is the player whose waste earns the most money.
The Game
The R-öko cards have a factory and a money side. They are sorted by colors and with the factory side up stacked in a
particular order. The exact composition of the stack depends on the number of players:
For 2 players the order from top to bottom is: 0, 1, 2, -2, 4, 5
For 3 or 4 players the order from top to bottom is: 0, 1, 2, 3, -2, 4, 5
With 5 players the order from top to bottom is: 0, 1, 2, 3, 3, -2, 4, 5
The four stacks with the R-öko factory cards are placed in the middle of the table side by side. The stacks represent
four recycling centers.
The “recycling” cards are shuffled and set as the draw pile in the middle of the table.
One “recycling” card is taken from the draw pile and added to the waste side of each recycling plant.
Now, all players take turns taking cards from the draw pile until everyone has three cards in hand. The player who
last took out the rubbish starts.
Draw Deck
(“recycling” cards)
Discard Pile
(“waste” cards)
Recycling
Side
Waste
Side
Factory
Cards
Game play
The players play in a clockwise direction. Each player completes a full turn before the next player begins.
(1) Play one or more “recycling” cards and take a R-öko factory card if appropriate
• The player recycles waste by playing one or more “recycling” cards of the appropriate type (colour) on
the recycling side of the factory.
• The player may play only “recycling” cards of one type (colour) in each turn.
• If the value of the “recycling” cards on the recycling side of the factory totals four or more, the player
receives the top R-öko factory card in the factory. To calculate this value, the cards already played and
the cards laid by the current player are added together. Note, "4" is not the number of cards, but their
total value: 1 “recycling” cards are single and 2 “recycling” cards count as double. The player only
receives one R-öko factory card, even if the total value of the rubbish cards "5", "6" or "10". The player
puts the R-öko factory card with the money side up and arranged by color in front of him. After all the
cards rubbish factory are placed in the discard pile.
• If the total value of the cards on the recycling side of the factory is "1", "2" or "3", the player receives no
R-öko factory card this turn.
(2) Take the “waste” cards
• The player must now take all the cards that are on the waste side of the same colored recycling factory
to their hand regardless of whether they received an R-öko factory card this turn or not
(3) Check hand limit
• Now the player must check how many cards he has in hand. They may have a maximum of five cards in
their hand. If the player has reached the limit, nothing happens, but if they have six or more cards in
hand, they must place the extra cards face down in front of them. These cards remain until the end of
the game are the players and are included in the final score at minus one point per card.
(4) Place “recycling” cards in the factory waste
• After all cards have been removed from the waste side of the factory, the player must refill the waste
pile by drawing a number of cards from the deck.
• The number of cards drawn is one more than the total value of the cards on the recycling side of the
factory.
• When the draw pile is exhausted, the discard pile is shuffled and recycled as a new draw pile.
Then the next player in a clockwise direction takes their turn
Game End
The game ends when a player takes the last R-öko factory card of one of the factories.
This player must complete their turn normally, including taking waste cards and checking their hand limit, discarding
any excess cards in the usual way. Now the R-öko factory cards and the face down cards of the players are
evaluated.
Scoring
• A player must have more than one R-öko factory card of a colour for it to be counted. If a player has only
one R-öko factory card of a given colour, this colour does not score points.
• Each player adds together the values of each colour of R-öko factory card.
• From this amount deduct any “recycling” cards face down in front of them (minus one point per card).
• Anyone with no face down “recycling” cards receives bonus points equal to the number of players with face
down “recycling” cards in front of them (plus one in four and five player games, plus two points for three
players, plus three for two players).
The player with the most points wins the game. In case of a tie, the player with the fewest hidden cards wins. If there
is still a tie, all players involved in the tie share the win.