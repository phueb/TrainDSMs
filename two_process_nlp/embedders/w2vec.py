import logging
from gensim.models import Word2Vec
import numpy as np

from two_process_nlp.embedders.base import EmbedderBase


class W2VecEmbedder(EmbedderBase):
    def __init__(self, param2val):
        super().__init__(param2val['param_name'], param2val['job_name'])
        self.w2vec_type = param2val['w2vec_type']
        self.embed_size = param2val['embed_size']
        self.window_size = param2val['window_size']
        self.num_epochs = param2val['num_epochs']
        #
        self.name = self.w2vec_type

    def train(self):
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        sg = Word2Vec(self.docs,
                      sg=True if self.w2vec_type == 'sg' else False,
                      size=self.embed_size,
                      window=self.window_size,
                      iter=self.num_epochs,
                      min_count=10,
                      workers=8,
                      hs=1)
        num_vocab = len(self.vocab)
        wvs = np.zeros((num_vocab, self.embed_size))
        for n, term in enumerate(self.vocab):
            term_acts = sg.wv[term]
            wvs[n] = term_acts
        self.w2e = self.embeds_to_w2e(wvs, self.vocab)
