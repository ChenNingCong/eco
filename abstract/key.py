"""
RNG key utilities — JAX-style deterministic seed management.

A Key wraps an integer seed. Use `key_from_seed` to create one from an int,
and `key.spawn(n)` to split into independent children.
"""

import random as _random


class Key:
    """Deterministic splittable key for RNG derivation."""

    __slots__ = ("seed",)

    def __init__(self, seed: int):
        self.seed = seed

    def spawn(self, n: int) -> list["Key"]:
        """Derive n independent child keys deterministically."""
        rng = _random.Random(self.seed)
        return [Key(rng.randrange(2**63)) for _ in range(n)]

    def make_rng(self) -> _random.Random:
        """Create a random.Random instance seeded from this key."""
        return _random.Random(self.seed)

    def __repr__(self) -> str:
        return f"Key({self.seed})"


def key_from_seed(seed: int) -> Key:
    """Create a Key from an integer seed."""
    return Key(seed)
