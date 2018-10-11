import numpy as np
from cytoolz import itertoolz
import pyprind


from src.utils import matrix_to_w2e
from src.embedders import EmbedderBase
from src import config

PAD = '*PAD*'
VERBOSE = False


class WWEmbedder(EmbedderBase):
    def __init__(self, corpus_name, name='ww'):
        super().__init__(corpus_name, name)

    @staticmethod
    def increment(mat, t1_id, t2_id, window_size, dist, window_weight):
        # check if PAD
        if t1_id == PAD or t2_id == PAD:
            #print(t1_id, t2_id)
            return
        # increment
        if window_weight == "linear":
            mat[t1_id, t2_id] += window_size - dist
        elif window_weight == "flat":
            mat[t1_id, t2_id] += 1
        if VERBOSE:
            print('Incrementing @ row {:>3} col {:>3}'.format(t1_id, t2_id))

    def update_matrix(self, mat, ids, window_size, window_weight):
        window_size = window_size + 1
        ids += [PAD] * window_size  # add padding such that all co-occurrences in last window are captured
        if VERBOSE:
            print(ids)
        windows = itertoolz.sliding_window(window_size, ids)
        for w in windows:
            for t1_id, t2_id, dist in zip([w[0]] * (window_size - 1), w[1:], range(1, window_size)):
                self.increment(mat, t1_id, t2_id, window_size, dist, window_weight)
        if VERBOSE:
            print()

    # TODO normalizations and reductions must be done before rturnign w2e
    def train(self, window_type=None, window_size=None, window_weight=None, norm_type=None, reduce_type=None, reduce_size=None):
        if window_type is None:
            window_type = config.WW.window_type
        if window_size is None:
            window_size = config.WW.window_size
        if window_weight is None:
            window_weight = config.WW.window_weight
        if norm_type is None:
            norm_type = config.Normalize.type
        if reduce_type is None:
            reduce_type = config.Reduce.type
        if reduce_size is None:
            reduce_size = config.Reduce.dimensions

        num_docs = len(self.numeric_docs)
        pbar = pyprind.ProgBar(num_docs)
        cooc_mat = np.zeros([config.Corpus.num_vocab, config.Corpus.num_vocab], int)
        print('\nCounting word-word co-occurrences in {}-word moving window'.format(config.WW.window_size))
        # count co-occurrences
        for token_ids in self.numeric_docs:
            self.update_matrix(cooc_mat, token_ids, window_size, window_weight)
            pbar.update()

        # co-occurrence type
        if window_type == 'forward':
            final_matrix = cooc_mat
        elif window_type == 'backward':
            final_matrix = cooc_mat.transpose()
        elif window_type == 'summed':
            final_matrix = cooc_mat + cooc_mat.transpose()
        elif window_type == 'concatenate':
            final_matrix = np.concatenate((cooc_mat, cooc_mat.transpose()))
        else:
            raise AttributeError("Improper matrix type '{}'. "
                                 "Must be 'forward', 'backward', 'summed', or 'concat'".format(config.WW.matrix_type))

        norm_matrix, dimensions = self.normalize(final_matrix, norm_type)
        reduced_matrix, dimensions = self.reduce(norm_matrix, reduce_type, reduce_size)

        w2e = matrix_to_w2e(reduced_matrix, self.vocab)
        return w2e
