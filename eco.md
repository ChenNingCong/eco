You have a repository of RL algorithms that run successfully on the heart-card game environment with self-play PPO.
Now your goal is to migrate the existing RL algorithms to the new R-öko game environment. The R-öko game has a different set of rules and mechanics compared to the heart-card game, the algorithms are still the same but the state space, action space, and reward structure will need to be adapted to fit the new game.
1. Read the game rule in eco_rule.md to understand the mechanics of the R-öko game.
2. Identify the state space, action space, and reward structure for the R-öko
    1. State Space: The state space for the R-öko game can be defined as a combination of the player's hand, the cards on the recycling and waste sides of the factory, and the R-öko factory cards collected by the player. This can be represented as a vector of integers where each element corresponds to a specific card or combination of cards. There are only two kinds of waster cards for every color, so the state space can be simplified by counting the number of each type of card rather than tracking individual cards. You need to represent the hands of each player, the discard piles, the draw pile, the factory sides, and the R-öko factory cards collected by each player. This can be done using a multi-dimensional array or a dictionary to keep track of the different components of the game state. (You may assume that this is a perfect information game, so the state space includes all information about the game that is available to the players.)
    2. Action Space: The action space for the R-öko game can be defined as the set of all possible actions a player can take during their turn.
       1. Playing one or more “recycling” cards of a specific color on the recycling side of the factory. The player can choose to play any combination of the cards in their hand, but they must all be of the same color (4 color x 5 x 5 (because there are two types card for each color))). Notice that you can play at most 5 cards in one turn, since the hand limit is 5 cards.
       2. Discarding excess cards if the player has more than five cards in hand after taking waste cards. The player can choose which cards to discard (8 possible combinations), but they must discard enough cards to bring their hand back down to five or fewer.
    3. So you can use a action phase token to represent the different phases of a player's turn (play cards or discard cards), if you need to discard multiple cards, just execute the discard action multiple times until the hand limit is satisfied.
    4. Reward Structure: can be calculated when a player gets the score token immediately!

For deck:
88 Recycling cards in four colours,
19 of them per colour with the value of
1 and 3 per colour with values of 2

Implement steps
1. Define the state space, action space, and reward structure for the R-öko game based on the rules provided in eco_rule.md.
2. Adapt the existing RL algorithms to work with the new state space, action space, and reward structure of the R-öko game. This may involve modifying the input and output layers of
3. Write a test suite to test the environment and the adapted RL algorithms to ensure they are working correctly with the new game mechanics. This can include unit tests for individual components of the environment, as well as integration tests to verify that the entire system is functioning as expected.
4. Run the adapted RL algorithms on the R-öko game environment and evaluate their performance. This can be done by measuring the average reward obtained by the algorithms over a number of episodes
5. modify the frontend server to support the new game environment and allow users to play against the RL agents. This may involve creating new user interfaces for the R-öko game, as well as integrating the RL agents into the existing frontend server architecture.
Please minimize the changes to the existing RL algorithms while ensuring that they can effectively learn and play the R-öko game, you should try to apply diff to the code instead of rewriting the entire algorithm from scratch. This will help to maintain the integrity of the original algorithms while allowing them to adapt to the new game environment.