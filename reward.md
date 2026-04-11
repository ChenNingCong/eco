I think you should reconsider the reward combination
middle game
1. delta(own)
2. delta(opp)
3. delta(own-opp)
combined with terminal reward of 
1. delta(own-opp) (note, this counts the last step difference, not the whole game difference) -> the total reward will be score differnece
2. -opp (sum to score difference)
3. any of the dense reward