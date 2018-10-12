import logging
from gensim.models import Word2Vec
import numpy as np

from src.embedders import EmbedderBase
from src import config
from src.utils import matrix_to_w2e


class W2VecEmbedder(EmbedderBase):
    def __init__(self, w2vec_type):
        super().__init__(w2vec_type)
        self.w2vec_type = w2vec_type
        assert w2vec_type in ['sg', 'cbow']

    def train(self):
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        sg = Word2Vec(self.numeric_docs,
                      sg=True if self.w2vec_type == 'sg' else False,
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
