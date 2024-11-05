from numpy.random import Generator

# The inclusive integer range for sampling seeds
SEED_RANGE: tuple[int, int] = (0, 1_000_000_000)

def generate_seed(rng: Generator) -> int:
    """Generates a random seed with the random generator."""
    return rng.integers(SEED_RANGE[0], SEED_RANGE[1])