import unittest
import numpy as np

from src import config
# this must be above CountEmbedder because it initializes load_corpus_data with config.Corpus.num_vocab
config.Corpus.name = 'ww_test'
config.Corpus.num_vocab = None

from src.params import gen_combinations, CountParams
from src.embedders.count import CountEmbedder


class MyTest(unittest.TestCase):
    def test_update_matrix(self):
        embedder = list(CountEmbedder(param2ids, param2val)
                        for param2ids, param2val in gen_combinations(CountParams))[0]
        embedder.name = 'ww'
        embedder.count_type = ['ww', 'backward', 5, 'flat']
        embedder.norm_type = None
        embedder.reduce_type = [None, None]
        embedder.param2val = {'count_type': embedder.count_type,
                              'norm_type': embedder.norm_type,
                              'reduce_type': embedder.reduce_type}
        #
        reduced_mat = embedder.train()
        correct = np.array([[0, 0, 2, 4, 3, 6],
                            [5, 0, 1, 3, 2, 4],
                            [0, 0, 0, 0, 0, 5],
                            [0, 0, 4, 0, 5, 3],
                            [0, 0, 5, 0, 0, 4],
                            [0, 0, 3, 5, 4, 2]])
        assert all(reduced_mat == correct)
        for i, j in zip(reduced_mat.flatten(), correct.flatten()):
            self.assertEqual(i, j)


if __name__ == '__main__':
    unittest.main()