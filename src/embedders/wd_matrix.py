import pyprind
import numpy as np

from src import config
from src.utils import matrix_to_w2e
from src.embedders import EmbedderBase


class WDEmbedder(EmbedderBase):

    def __init__(self, corpus_name, name='wd'):
        super().__init__(corpus_name, name)

    def train(self, norm_type=None, reduce_type=None, reduce_size=None):
        if norm_type is None:
            norm_type = config.Normalize.type
        if reduce_type is None:
            reduce_type = config.Reduce.type
        if reduce_size is None:
            reduce_size = config.Reduce.dimensions

        num_docs = len(self.numeric_docs)
        pbar = pyprind.ProgBar(num_docs)
        count_matrix = np.zeros([config.Corpus.num_vocab, num_docs], int)
        print('\nCounting word occurrences in {} documents'.format(num_docs))
        for i in range(num_docs):
            for token_id in self.numeric_docs[i]:
                count_matrix[token_id,i] += 1
            pbar.update()

        norm_matrix, dimensions = self.normalize(count_matrix, norm_type)
        reduced_matrix, dimensions = self.reduce(norm_matrix, reduce_type, reduce_size)

        w2e = matrix_to_w2e(reduced_matrix, self.vocab)

        return w2e