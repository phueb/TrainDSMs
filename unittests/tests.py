import unittest
import numpy as np

from src import config
from src.embedders.count import CountEmbedder


class MyTest(unittest.TestCase):
    def test_update_matrix(self):
        config.WW.window_size = 4
        config.WW.window_weight = 'flat'
        config.WW.window_type = 'summed'
        embedder = CountEmbedder('test_corpus')
        config.Corpus.num_vocab = len(embedder.vocab)
        cooc_mat = np.zeros([config.Corpus.num_vocab, config.Corpus.num_vocab], int)
        for token_ids in embedder.numeric_docs:
            embedder.update_matrix(cooc_mat, token_ids, config.WW.window_size,config.WW.window_weight)
        # assertEqual
        correct = np.load('unittests/test_corpus_co-oc_mat.npy')
        for i, j in zip(cooc_mat.flatten(), correct.flatten()):
            self.assertEqual(i, j)


if __name__ == '__main__':
    unittest.main()