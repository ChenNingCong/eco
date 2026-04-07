 I believe the real problem is that the action space is larger than model's hidden dim, and may we should also take the action mask as input
  can you decompose the action into a sequential one and make also take the action mask as input to make it explicit for the model?
Maybe we can follow the design of r-eco
We also use a phase marker as input? The action space is the concatenate of
1. Card id
2. Draw or discard (1 + 5 card)
So we firstly choose a card, then choose whether to draw from deck or draw from discard pile (notice that we must mask out the illegal actions)
Also run ablation for maxlanes=4/5 and 50M steps