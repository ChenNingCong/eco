Maybe it's a problem of network architecture, let's run ablation here (use flag)
Note : can you use flags with smart names, for example --network.XXX
1. Stablize LSTM
   1. shift forget gate (less likely, because I really think even if we ignore history it should be strong)
   2. clip norm -> maybe network is unstable
