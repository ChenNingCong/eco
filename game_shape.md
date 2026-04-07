Let's think of one problem:
Maybe it's not that our algorithm is suboptimal, it's the game itself that is suboptimal.
Notice that build long routes is much sampler than completing destinations and env is of high variance.
So I think the problem is that the scores for destination are too low.
Let's run ablation here, we add a parameter called scale_route_score, it scales the score of routers by s let's see what happens (we here should use s < 1).
If the model learns to increase the destination_completed then it's good
Another choose is to reduce penalty let's introduce another parameter, we should scales down the penaltly (but still the reward is the same)
For example, complete one destination gets 10 scores but fails to complete it only loses 10 * s scores
You wandb should still report the unmodified game store.
To verify this, let's start with some extreme exp choices - maybe s = 0.1
Use entropy from 0.1 to 0.01 in 1M steps with exp scheduler as baseline, 10M steps