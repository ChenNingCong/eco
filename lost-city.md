https://github.com/ChenNingCong/Lost-City.git
now port the lost city game from my previos game with the following modifications:
1. You can reuse the hyperparameter for eco and ttr (also 10M samples, 1M with entropy 0.1 -> 0.01 exp decay)
2. Benchmark: random and self-play, show original score
   1. Add additional metric for: number of investment cards (to see whether the AI prefers long term goal), number of played cards, number of open lanes
3. Add game adjustable parameter:
   1. baseline: currently when a player plays a card in a new color line, it always get penalty of -20
   2. Run ablations for -15 and -10
Run all of the code by yourself