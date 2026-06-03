Let's fuse all the step into one loop
it's like this (it can be multi-threaded inside numba?)
step() -> numba jit function:   
    for each environment run step and collect obs
    for each environment that's in another player's round
        run pytorch batched inference using objmode
