import numpy as np
from cytoolz import itertoolz
import pyprind
from sortedcontainers import SortedDict

from src.embedders import EmbedderBase
from src import config

PAD = '*PAD*'


class WWEmbedder(EmbedderBase):
    def __init__(self, corpus_name):
        super().__init__(corpus_name, 'ww')

    @staticmethod
    def increment(mat, t1_id, t2_id, window_size, dist):
        # check if PAD
        if t1_id == PAD or t2_id == PAD:
            print(t1_id, t2_id)
            return
        # increment
        if config.WW.window_weight == "linear":
            mat[t1_id, t2_id] += window_size - dist
        elif config.WW.window_weight == "flat":
            mat[t1_id, t2_id] += 1
        print('Incrementing @ row {:>3} col {:>3}'.format(t1_id, t2_id))

    def update_matrix(self, mat, ids):
        window_size = config.WW.window_size + 1
        ids += [PAD] * window_size  # add padding such that all co-occurrences in last window are captured
        print(ids)
        windows = itertoolz.sliding_window(window_size, ids)
        for w in windows:
            for t1_id, t2_id, dist in zip([w[0]] * (window_size - 1), w[1:], range(1, window_size)):
                self.increment(mat, t1_id, t2_id, window_size, dist)
            print()

    # TODO this should return nromalized + reduced matrix
    # TODO ideally use 1 matrix embedder class and move all relevanta functions from EmbedderBase
    def train(self):
        num_docs = len(self.numeric_docs)

        pbar = pyprind.ProgBar(num_docs)
        cooc_mat = np.zeros([config.Corpora.num_vocab, config.Corpora.num_vocab], int)
        print('\nCounting word-word co-occurrences in {}-word moving window'.format(config.WW.window_size))
        for token_ids in self.numeric_docs:
            self.update_matrix(cooc_mat, token_ids)
            pbar.update()

        if config.WW.matrix_type == 'forward':
            final_matrix = cooc_mat
        elif config.WW.matrix_type == 'backward':
            final_matrix = cooc_mat.transpose()
        elif config.WW.matrix_type == 'summed':
            final_matrix = cooc_mat + cooc_mat.transpose()
        elif config.WW.matrix_type == 'concatenate':
            final_matrix = np.concatenate((cooc_mat, cooc_mat.transpose()))
        else:
            raise AttributeError("Improper matrix type '{}'. "
                                 "Must be 'forward', 'backward', 'summed', or 'concat'".format(config.WW.matrix_type))
        w2e = SortedDict({w: final_matrix[n] for n, w in enumerate(self.vocab)})
        return w2e

    def verify(self, input_matrix):  # TODO
        assert config.WW.window_weight == 'flat'  # only works when using "flat"
        num_windows = 0
        for token_ids in self.numeric_docs:
            num_windows += (len(token_ids) - config.WW.window_size + 1)
        num_coocs_per_window = config.WW.window_size - 1
        #assert np.sum(cooc_mat) == (num_windows * num_coocs_per_window)
