import logging
from gensim.models import Word2Vec
import numpy as np

from src.embedders.base import EmbedderBase
from src.params import Word2VecParams
from src.utils import matrix_to_w2e


class W2VecEmbedder(EmbedderBase):
    def __init__(self, param2ids):
        super().__init__()
        self.w2vec_type = Word2VecParams.w2vec_type[param2ids.w2vec_type]
        self.embed_size = Word2VecParams.embed_size[param2ids.w2vec_type]
        self.window_size = Word2VecParams.window_size[param2ids.window_size]
        self.num_epochs = Word2VecParams.num_epochs[param2ids.num_epochs]
        #
        self.name = self.w2vec_type

    def train(self):
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        sg = Word2Vec(self.numeric_docs,
                      sg=True if self.w2vec_type == 'sg' else False,
                      size=self.embed_size,
                      window=self.window_size,
                      iter=self.num_epochs,
                      min_count=10,
                      workers=8,
                      hs=1)
        num_vocab = len(self.vocab)
        embed_mat = np.zeros((num_vocab, self.embed_size))
        for n, term in enumerate(self.vocab):
            term_acts = sg.wv[term]
            embed_mat[n] = term_acts
        w2e = matrix_to_w2e(embed_mat, self.vocab)
        return w2e
