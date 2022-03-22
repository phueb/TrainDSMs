import numpy as np
from typing import List, Dict
from typing import Tuple


from traindsms.params import RandomControlParams


class RandomControlDSM:
    def __init__(self,
                 params: RandomControlParams,
                 vocab: Tuple[str],
                 ):
        self.params = params
        self.vocab = vocab

        self.t2e = None

    def train(self):
        if self.params.distribution == 'normal':
            self.t2e = {w: np.random.normal(0, 1.0, self.params.embed_size) for w in self.vocab}
        elif self.params.distribution == 'uniform':
            self.t2e = {w: np.random.uniform(-1.0, 1.0, self.params. embed_size) for w in self.vocab}
        else:
            raise NotImplementedError

    def get_performance(self) -> Dict:
        return {}

    def calc_native_sr_scores(self,
                              verb: str,
                              theme: str,
                              instruments: List[str],
                              ) -> List[float]:

        """
        this is one of two way sof computing random scores. this is the 'native' method.
        alternatively, one can use the random embeddings in the vector composition task.

        by default, composition_fn=native, therefore this method is used.
        """
        verb = verb
        theme = theme

        # get random scores
        scores = np.random.uniform(-1, +1, len(instruments)).tolist()

        return scores
