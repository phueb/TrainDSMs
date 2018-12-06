import numpy as np

from src.embedders.base import EmbedderBase
from src.params import RandomControlParams


class RandomControlEmbedder(EmbedderBase):
    def __init__(self, param2ids, param2val):
        super().__init__(param2val)
        self.embed_size = RandomControlParams.embed_size[param2ids.embed_size]
        self.random_type = RandomControlParams.random_type[param2ids.random_type]
        #
        self.name = 'random_control'

    def train(self):
        if self.random_type == 'normal':
            self.w2e = {w: np.random.normal(0, 1.0, self.embed_size) for w in self.vocab}
        elif self.random_type == 'uniform':
            self.w2e = {w: np.random.uniform(-1.0, 1.0, self.embed_size) for w in self.vocab}
        else:
            raise NotImplementedError
