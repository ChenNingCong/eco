Let's image a real vectorized environment implementation! Currently the game loop is like this:
def step(self, action):
    # current player takes action
    # update game state based on action
    if necessary:
        for opponent in opponents:
            # opponent takes action <- evaluate neural network for opponent's action
            # update game state based on opponent's action
And the vectorized environment implementation would look like this:
def step(self, actions):
    for game in games:
        game.step(actions[game.id])  # each game processes its own action
The problem is that the neural network evaluation for opponents' actions is currently intertwined with the game loop, which makes it difficult to separate the two and implement a vectorized environment. To achieve a vectorized implementation, we need to decouple the neural network evaluation from the game loop and allow each game to process its own actions independently. This would involve restructuring the code to ensure that each game can handle its own state updates and action processing without relying on a centralized loop that evaluates opponents' actions.

Currently my thought is to use generator function like this:
For single game loop:
def step(self, action):
    # current player takes action
    # update game state based on action
    if necessary:
        for opponent in opponents:
            yield opponent_observation  # yield opponent's observation for neural network evaluation
            # then action is sent back to the game loop after neural network evaluation
            # update game state based on opponent's action
def step(self, actions):
    generators = [game.step(actions[game.id]) for game in games]
    # try to exhaust the generators until all games have processed their actions
    masks = [True] * len(games)  # mask to track which generators are still active
    def one_round()
        current_observations = []
        # first step: collect opponent observations from all active generators
        for i, gen in enumerate(generators):
            if masks[i]:  # if this generator is still active
                try:
                    opponent_observation = next(gen)  # get the next opponent observation
                    current_observations.append(opponent_observation)
                except StopIteration:
                    masks[i] = False  # this generator has finished processing
        # second step: evaluate neural network for all collected opponent observations
        opponent_actions = neural_network_evaluation(torch.stack(current_observations))
        # third step: send opponent actions back to the generators
        for i, gen in enumerate(generators):
            if masks[i]:  # if this generator is still active
                try:
                    gen.send(opponent_actions[i])  # send the corresponding opponent action back to the generator
                except StopIteration:
                    masks[i] = False  # this generator has finished processing
    # keep running one_round until all generators have finished processing
    while any(masks):
        one_round()
This way, each game can independently process its own actions and update its state, while the neural network evaluation for opponents' actions is handled in a separate step that collects observations from all active games, evaluates the neural network in a batch, and then sends the corresponding actions back to each game. This decouples the neural network evaluation from the game loop and allows for a vectorized implementation of the environment.    