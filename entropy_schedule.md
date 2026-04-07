0.1 entropy is too large and 0.01 is too small (but still fine)
Let's still use 1-0.01 schedule but run for 50% of the time (5M samples), create script and run it
Can you think of a better scheduler, linear seems to be ineffective 
For example we always use 1, 0.1, 0.01 (logspace) - research the literature to understand which scheduler is better