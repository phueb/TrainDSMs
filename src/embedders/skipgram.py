import logging
from gensim.models import Word2Vec
import numpy as np

from src.embedders import EmbedderBase
from src import config
from src.utils import matrix_to_w2e


class WDEmbedder(EmbedderBase):
    def __init__(self, corpus_name):
        super().__init__(corpus_name, 'wd')

    def train(self):
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        sg = Word2Vec(self.numeric_docs,
                      sg=1,
                      size=config.Skipgram.embed_size,
                      window=config.Skipgram.window_size,
                      iter=config.Skipgram.num_epochs,
                      min_count=10,
                      workers=8,
                      hs=1)
        num_vocab = len(self.vocab)
        embed_mat = np.zeros((num_vocab, config.Skipgram.embed_size))
        for n, term in enumerate(self.vocab):
            term_acts = sg.wv[term]
            embed_mat[n] = term_acts
        w2e = matrix_to_w2e(embed_mat, self.vocab)
        return w2e
