import pyprind
import numpy as np

from src import config
from src.utils import matrix_to_w2e
from src.embedders import EmbedderBase


class WDEmbedder(EmbedderBase):
    def __init__(self, corpus_name):
        super().__init__(corpus_name, 'wd')

    def train(self):
        num_docs = len(self.numeric_docs)
        pbar = pyprind.ProgBar(num_docs)
        count_matrix = np.zeros([config.Corpora.num_vocab, num_docs], int)
        print('\nCounting word occurrences in {} documents'.format(num_docs))
        for i in range(num_docs):
            for token_id in self.numeric_docs[i]:
                count_matrix[token_id,i] += 1
            pbar.update()
        w2e = matrix_to_w2e(count_matrix, self.vocab)
        return w2e