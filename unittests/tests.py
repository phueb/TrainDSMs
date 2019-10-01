import unittest
import numpy as np

from two_process_nlp import config
from two_process_nlp.job import preprocessing_job
from two_process_nlp.params import CountParams
from two_process_nlp.embedders.count import CountEmbedder

from ludwigcluster.utils import list_all_param2vals


class MyTest(unittest.TestCase):
    def test_update_matrix(self):
        #
        config.Corpus.name = 'ww_test'
        config.Corpus.num_vocab = None
        preprocessing_job()
        # embedder
        embedder = list(CountEmbedder(param2val)
                        for param2val in list_all_param2vals(CountParams))[0]
        embedder.name = 'ww'
        embedder.count_type = ['ww', 'backward', 5, 'linear']
        embedder.norm_type = None
        embedder.reduce_type = [None, None]
        #
        reduced_mat = embedder.train()
        correct = np.array([[0, 0, 2, 4, 3, 6],
                            [5, 0, 1, 3, 2, 4],
                            [0, 0, 0, 0, 0, 5],
                            [0, 0, 4, 0, 5, 3],
                            [0, 0, 5, 0, 0, 4],
                            [0, 0, 3, 5, 4, 2]])
        for i, j in zip(reduced_mat.flatten(), correct.flatten()):
            self.assertEqual(i, j)


if __name__ == '__main__':
    unittest.main()