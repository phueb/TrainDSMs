import numpy as np

from src.embedders import EmbedderBase


class RandomControlEmbedder(EmbedderBase):
    def __init__(self, corpus_name):
        super().__init__(corpus_name, 'random_control')

    def train(self):
        embed_size = 512
        w2e = {w: np.random.normal(0, 1.0, embed_size) for n, w in enumerate(self.vocab)}
        return w2e, embed_size