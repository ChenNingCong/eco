"""
RNG key utilities.

Key is just np.random.Generator (backed by PCG64 + SeedSequence).
Use key_from_seed(int) to create one, and key.spawn(n) to split into
independent children. Generator.spawn() (NumPy ≥1.25) produces
statistically independent streams via SeedSequence splitting.
"""

import numpy as np

Key = np.random.Generator


def key_from_seed(seed: int) -> Key:
    """Create a Key (numpy Generator) from an integer seed."""
    return np.random.default_rng(seed)
