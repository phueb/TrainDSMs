import numpy as np

from src.embedders.base import EmbedderBase
from src.params import RandomControlParams


class RandomControlEmbedder(EmbedderBase):
    def __init__(self, param2ids, param2val):
        super().__init__(param2val)
        self.distribution = RandomControlParams.distribution[param2ids.distribution]
        self.embed_size = RandomControlParams.embed_size[param2ids.embed_size]
        #
        self.name = 'random_control'

    def train(self):
        if self.distribution == 'uniform':
            w2e = {w: np.random.uniform(-1.0, 1.0, self.embed_size) for n, w in enumerate(self.vocab)}
        elif self.distribution == 'normal':
            w2e = {w: np.random.normal(0, 1.0, self.embed_size) for n, w in enumerate(self.vocab)}
        else:
            raise AttributeError('Invalid arg to RandomControl.distribution.')
        return w2e, self.embed_size
