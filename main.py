from environment import GamblerGame

TARGET_WEALTH: int = 20
WIN_PROB: float = 0.5
SEED: int = 0

env = GamblerGame(TARGET_WEALTH, WIN_PROB, SEED)
state = env.reset()