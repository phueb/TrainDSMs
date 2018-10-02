import numpy as np
from cytoolz import itertoolz
import pyprind

from src.embedders import EmbedderBase
from src import config

class WWEmbedder(EmbedderBase):
    def __init__(self, corpus_name):
        super().__init__(corpus_name, 'hal')

    def update_matrix(self, mat, ids):
        window_size = config.WW.window_size + 1
        windows = itertoolz.sliding_window(window_size, ids)
        for w in windows:
            for t1_id, t2_id, dist in zip([w[0]] * (window_size - 1), w[1:], range(1, window_size)):
                if config.WW.window_weight == "linear":
                    mat[t1_id, t2_id] += window_size - dist
                elif config.WW.window_weight == "flat":
                    mat[t1_id, t2_id] += 1

    def train(self):
        num_docs = len(self.numeric_docs)

        pbar = pyprind.ProgBar(num_docs)
        cooc_mat = np.zeros([config.Corpora.num_vocab, config.Corpora.num_vocab], int)
        for token_ids in self.numeric_docs:
            self.update_matrix(cooc_mat, token_ids)
            pbar.update()
        w2e = {w: cooc_mat[n] for n, w in enumerate(self.vocab)}
        embed_size = cooc_mat.shape[1]  # TODO add normalizations and reductions here
        self.verify(cooc_mat)

        return w2e, embed_size

    def verify(self, cooc_mat):
        assert config.WW.window_weight == 'flat'  # only works when using "flat"
        num_windows = 0
        for token_ids in self.numeric_docs:
            num_windows += (len(token_ids) - config.WW.window_size + 1)
        num_coocs_per_window = config.WW.window_size - 1
        #assert np.sum(cooc_mat) == (num_windows * num_coocs_per_window)
