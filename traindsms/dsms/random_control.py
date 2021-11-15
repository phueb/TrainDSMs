import numpy as np
from typing import Tuple, List


from traindsms.params import RandomControlParams


class RandomControlDSM:
    def __init__(self,
                 params: RandomControlParams,
                 vocab: Tuple[str],
                 ):
        self.params = params
        self.vocab = vocab

        self.t2e = None

    def train(self):
        if self.params.random_type == 'normal':
            self.t2e = {w: np.random.normal(0, 1.0, self.params.embed_size) for w in self.vocab}
        elif self.params.random_type == 'uniform':
            self.t2e = {w: np.random.uniform(-1.0, 1.0, self.params. embed_size) for w in self.vocab}
        else:
            raise NotImplementedError
