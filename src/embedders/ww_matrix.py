import numpy as np
from cytoolz import itertoolz
import pyprind


from src.utils import matrix_to_w2e
from src.embedders import EmbedderBase
from src import config

PAD = '*PAD*'
VERBOSE = False


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
        if VERBOSE:
            print('Incrementing @ row {:>3} col {:>3}'.format(t1_id, t2_id))

    def update_matrix(self, mat, ids):
        window_size = config.WW.window_size + 1
        ids += [PAD] * window_size  # add padding such that all co-occurrences in last window are captured
        if VERBOSE:
            print(ids)
        windows = itertoolz.sliding_window(window_size, ids)
        for w in windows:
            for t1_id, t2_id, dist in zip([w[0]] * (window_size - 1), w[1:], range(1, window_size)):
                self.increment(mat, t1_id, t2_id, window_size, dist)
        if VERBOSE:
            print()

    # TODO normalizations and reductions must be done before rturnign w2e
    def train(self):
        num_docs = len(self.numeric_docs)
        pbar = pyprind.ProgBar(num_docs)
        cooc_mat = np.zeros([config.Corpora.num_vocab, config.Corpora.num_vocab], int)
        print('\nCounting word-word co-occurrences in {}-word moving window'.format(config.WW.window_size))
        # count co-occurrences
        for token_ids in self.numeric_docs:
            self.update_matrix(cooc_mat, token_ids)
            pbar.update()
        # co-occurrence type
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
        w2e = matrix_to_w2e(final_matrix, self.vocab)
        return w2e
