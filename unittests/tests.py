import unittest
import numpy as np

from traindsms.params import CountParams
from traindsms.dsms.count import CountDSM


class MyTest(unittest.TestCase):
    def test_update_matrix(self):

        doc0 = 'the horse raced past the barn fell'.split()
        vocab = sorted(set(doc0))
        token2id = {t: n for n, t in enumerate(vocab)}
        numeric_docs = []
        for doc in [doc0]:
            numeric_docs.append([token2id[token] for token in doc])

        param2val = {
            'count_type': ['ww', 'backward', 5, 'linear'],
            'norm_type': None,
            'reduce_type': [None, None],
        }
        params = CountParams.from_param2val(param2val)
        dsm = CountDSM(params, numeric_docs)

        reduced_mat = dsm.train()
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
