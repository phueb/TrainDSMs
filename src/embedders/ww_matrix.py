import numpy as np
from cytoolz import itertoolz
import pyprind

from src.embedders import EmbedderBase
from src import config

class WWEmbedder(EmbedderBase):
    def __init__(self, corpus_name):
        super().__init__(corpus_name, 'ww')

    def update_matrix(self, mat, ids):
        window_size = config.WW.window_size + 1
        windows = itertoolz.sliding_window(window_size, ids)
        for w in windows:
            #print(self.vocab[w[0]], self.vocab[w[1]], self.vocab[w[2]])
            for t1_id, t2_id, dist in zip([w[0]] * (window_size - 1), w[1:], range(1, window_size)):
                if config.WW.window_weight == "linear":
                    mat[t1_id, t2_id] += window_size - dist
                elif config.WW.window_weight == "flat":
                    mat[t1_id, t2_id] += 1
        # count the final windows that are smaller than window_size
        while(len(w) > 1):
            t1_id = w[0]
            final_window = w[1:]
            for i in range(len(final_window)):
                t2_id = final_window[i]
                if config.WW.window_weight == "linear":
                    mat[t1_id, t2_id] += window_size - i
                elif config.WW.window_weight == "flat":
                    mat[t1_id, t2_id] += 1
            w = w[1:]

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
            print("Improper matrix type '{}'. Must be 'forward', 'backward', 'summed', or 'concat'".format(config.WW.matrix_type))
            sys.exit(2)

        w2e = {w: final_matrix[n] for n, w in enumerate(self.vocab)}
        embed_size = final_matrix.shape[1]
        #self.verify(cooc_mat) # TODO verify function is now broken in several ways due to window size and matrix type

        return w2e, embed_size

    def verify(self, cooc_mat):
        assert config.WW.window_weight == 'flat'  # only works when using "flat"
        num_windows = 0
        for token_ids in self.numeric_docs:
            num_windows += (len(token_ids) - config.WW.window_size + 1)
        num_coocs_per_window = config.WW.window_size - 1
        #assert np.sum(cooc_mat) == (num_windows * num_coocs_per_window)
