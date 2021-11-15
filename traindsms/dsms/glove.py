

from typing import List, Tuple

from traindsms.params import GloveParams


class GloVe:
    def __init__(self,
                 params: GloveParams,
                 vocab: Tuple[str],
                 seq_tok: List[List[str]],
                 ):

        self.params = params
        self.vocab = vocab
        self.seq_tok = seq_tok

        self.t2e = None

    def train(self):
        return NotImplementedError('Use original implementation')

        # todo https://github.com/pengyan510/nlp-paper-implementation/blob/master/glove/src/train.py
