something bad happens! When we switch to LSTM, we encounter the same problem as gae and entropy
The model just gets stuck! (the reward level and win rate is the same)
Difference: explained variance is high, though slightly worse than 10M exp
entorpy is also high
kl is pretty normal, what could be the problem?
Hypothesis
1. Maybe it's because gradient explosion or gradient vanishing?
   1. Add metrics to measure gradient (i.g. monitor RNN training dynamics)
2. Let's add a flag to turn LSTM into non-LSTM -> we ignore the history during caculation then rerun exp to see what happened
3. Entropy still too low, let entropy last longer and larger
4. Strategy coadaption -> the model uses LSTM to coadapt with the opponent, which is bad
   1. (We test this later)