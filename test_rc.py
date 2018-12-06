import numpy as np

from src import config


EMBED_SIZE = 30

w2e = {w: np.random.normal(0, 1.0, EMBED_SIZE) for w in self.vocab}