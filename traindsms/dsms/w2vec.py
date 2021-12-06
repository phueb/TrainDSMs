import logging
from gensim.models import Word2Vec
from typing import Tuple, List
import numpy as np

from traindsms.params import Word2VecParams


class W2Vec:
    def __init__(self,
                 params: Word2VecParams,
                 vocab: Tuple[str],
                 seq_tok: List[List[str]],
                 ):

        self.params = params
        self.vocab = vocab
        self.seq_tok = seq_tok

        self.t2e = None

    def train(self):
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        sg = Word2Vec(sentences=self.seq_tok,
                      sg=True if self.params.w2vec_type == 'sg' else False,
                      vector_size=self.params.embed_size,
                      window=self.params.window_size,
                      epochs=self.params.num_epochs,
                      min_count=1,
                      workers=1,
                      hs=1)

        self.t2e = {t: np.asarray(sg.wv[t]) for t in self.vocab}

    def get_performance(self):
        return {}
