import numpy as np

from src.embedders import EmbedderBase
from src import config


class RandomControlEmbedder(EmbedderBase):
    def __init__(self):
        super().__init__('random_control')

    def train(self):
        if config.RandomControl.distribution == 'uniform':
            w2e = {w: np.random.uniform(-1.0, 1.0, config.RandomControl.embed_size) for n, w in enumerate(self.vocab)}
        elif config.RandomControl.distribution == 'normal':
            w2e = {w: np.random.normal(0, 1.0, config.RandomControl.embed_size) for n, w in enumerate(self.vocab)}
        else:
            raise AttributeError('Invalid arg to RandomControl.distribution.')
        return w2e, config.RandomControl.embed_size
