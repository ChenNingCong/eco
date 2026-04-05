Currently everything it's typically coupled to a single kind of game (even the name it's set to eco)
in a folder abstract/ (we will rename it later), let's create a general trainer and definition that works across game
Firstly think about the general architecture! Here is my thought
1. An abstract game engine interface that supports an interface for game logic <- This is customizable
   `step(obs, action, player_id) -> next_player_id, obs, ...` which allows current player to perform one action
   Note: the reward is now an array (for each of the player)
2. Some game wrappers that transform the game (reward shaping or whatever. similar to gymnasium)
3. Self play environment (`SinglePlayerEcoEnv` in our current implementation)
    `step(obs, action, player_id) -> obs, ...`
    only allows one player to act (other players are part of the environment)
    This is only partially customizable - seat randomization is also general, but maybe sometimes the user wants to do some magic
    let's ignore this and assume that this is not customizable
4. Vectorized self play environment -> not customizable
5. Multiprocessing envrionment -> not customizable

BatchedPlayer, SlicedPlayer, Offset they are the same
Note: I think a better design is to take factory as input to construct environment, each factory gets the id of current env (and a seed)
Otherwise Multiprocessing envrionment and Vectorized self play environment they need to pass configs down to the game engine constructor

Let's firstly focus on the interface, not the game loop