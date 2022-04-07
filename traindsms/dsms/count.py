import numpy as np
from cytoolz import itertoolz
import pyprind
import sys
import time
from typing import List, Tuple

from traindsms.params import CountParams

PAD = '*PAD*'
VERBOSE = False


class CountDSM:
    def __init__(self,
                 params: CountParams,
                 vocab: Tuple[str],
                 seq_num: List[List[int]],
                 ):
        self.params = params
        self.vocab = vocab
        self.seq_num = seq_num
        self.vocab_size = len(vocab)

        self.t2e = None

    # ////////////////////////////////////////////////// word-by-word

    def create_ww_matrix_fast(self):  # no function call overhead - twice as fast
        window_type = self.params.count_type[1]
        window_size = self.params.count_type[2]
        window_weight = self.params.count_type[3]

        print('Counting word-word co-occurrences in {}-word moving window'.format(window_size))

        # count
        num_docs = len(self.seq_num)
        pbar = pyprind.ProgBar(num_docs, stream=sys.stdout)
        count_matrix = np.zeros([self.vocab_size, self.vocab_size], int)
        for token_ids in self.seq_num:
            token_ids += [PAD] * window_size  # add padding such that all co-occurrences in last window are captured

            if VERBOSE:
                print(token_ids)

            for w in itertoolz.sliding_window(window_size + 1, token_ids):  # + 1 because window consists of t2s only

                if VERBOSE:
                    print([self.vocab[i] if isinstance(i, int) else PAD for i in w])

                # in a window, we count all co-occurrences between first token and all others.
                # a co-occurrence is between a t1_id and a t2_id, each corresponds to the ID of a token in the vocab.

                t1_id = w[0]
                for dist, t2_id in enumerate(w[1:]):

                    # increment
                    if t1_id == PAD or t2_id == PAD:
                        continue
                    if window_weight == "linear":
                        count_matrix[t1_id, t2_id] += window_size - dist
                    elif window_weight == "flat":
                        count_matrix[t1_id, t2_id] += 1
                    if VERBOSE:
                        print('row {:>3} col {:>3} set to {}'.format(t1_id, t2_id, count_matrix[t1_id, t2_id]))

            if VERBOSE:
                print()
            pbar.update()

        # window_type
        if window_type == 'forward':
            final_matrix = count_matrix
        elif window_type == 'backward':
            final_matrix = count_matrix.transpose()
        elif window_type == 'summed':
            final_matrix = count_matrix + count_matrix.transpose()
        elif window_type == 'concatenated':
            final_matrix = np.concatenate((count_matrix, count_matrix.transpose()), axis=1)
        else:
            raise AttributeError('Invalid arg to "window_type".')

        print('Shape of normalized matrix={}'.format(final_matrix.shape))

        return final_matrix

    # ////////////////////////////////////////////////// word-by-document

    def create_wd_matrix(self):
        num_docs = len(self.seq_num)
        pbar = pyprind.ProgBar(num_docs, stream=sys.stdout)
        count_matrix = np.zeros([self.vocab_size, num_docs], int)
        print('\nCounting word occurrences in {} documents'.format(num_docs))
        for i in range(num_docs):
            for j in self.seq_num[i]:
                count_matrix[j, i] += 1
            pbar.update()
        return count_matrix

    # ////////////////////////////////////////////////// train

    def train(self):
        # count
        start = time.time()
        if self.params.count_type[0] == 'ww':
            count_matrix = self.create_ww_matrix_fast()
        elif self.params.count_type[0] == 'wd':
            count_matrix = self.create_wd_matrix()
        else:
            raise AttributeError('Invalid arg to "count_type".')

        print(f'Completed count in {time.time() - start}', flush=True)

        # normalize + reduce
        norm_matrix = normalize(count_matrix, self.params.norm_type)
        reduced_matrix = reduce(norm_matrix, self.params.reduce_type[0], self.params.reduce_type[1])

        self.t2e = {t: e for t, e in zip(self.vocab, reduced_matrix)}

        return reduced_matrix  # for unittest

    def get_performance(self):
        return {}

# ////////////////////////////////////////////////// normalizations


def normalize(input_matrix: np.array,
              norm_type: str,
              ) -> np.array:
    if norm_type == 'row_sum':
        norm_matrix = norm_rowsum(input_matrix)
    elif norm_type == 'col_sum':
        norm_matrix = norm_col_sum(input_matrix)
    elif norm_type == 'tf_idf':
        norm_matrix = norm_tfidf(input_matrix)
    elif norm_type == 'row_logentropy':
        norm_matrix = row_log_entropy(input_matrix)
    elif norm_type == 'ppmi':
        norm_matrix = norm_ppmi(input_matrix)
    elif norm_type is None:
        norm_matrix = input_matrix
    else:
        raise AttributeError(f"Improper matrix normalization type '{norm_type}'. "
                             "Must be 'row_sum', 'col_sum', 'row_logentropy', 'tf-idf', 'ppmi', or 'none'")
    return norm_matrix


def norm_rowsum(input_matrix: np.array,
                ) -> np.array:
    print('Normalizing matrix by row sums')

    num_rows = input_matrix.shape[0]
    res = np.zeros_like(input_matrix, float)
    for i in range(num_rows):
        if input_matrix[i, :].sum() == 0:
            print('    Warning: Row {} had sum of zero. Setting prob to 0'.format(i))
        else:
            res[i, :] = input_matrix[i, :] / input_matrix[i, :].sum()
    return res


def norm_col_sum(input_matrix: np.array,
                 ) -> np.array:
    print('Normalizing matrix by column sums')

    num_cols = input_matrix.shape[1]
    res = np.zeros_like(input_matrix, float)
    for i in range(num_cols):
        if input_matrix[:, i].sum() == 0:
            print('    Warning: Column {} had sum of zero. Setting prob to 0'.format(i))
        else:
            res[:, i] = input_matrix[:, i] / input_matrix[:, i].sum()
    return res


def norm_tfidf(input_matrix: np.array,
               ) -> np.array:
    print('Normalizing matrix by tf-idf')
    
    num_rows = input_matrix.shape[0]
    num_cols = nd = input_matrix.shape[1]

    res = np.zeros_like(input_matrix, float)
    for i in range(num_rows):
        df = np.count_nonzero(input_matrix[i, :])  # num documents/columns in which word (in row) occurs
        idf = np.log((nd + 1) / (df + 1)) + 1
        for j in range(num_cols):
            tf = np.log(1 + res[i, j])
            res[i, j] = tf * idf
    return res


def norm_ppmi(input_matrix: np.array,
              ) -> np.array:
    print('Normalizing matrix by ppmi')

    num_rows = input_matrix.shape[0]
    num_cols = input_matrix.shape[1]

    row_sums = input_matrix.sum(1)
    col_sums = input_matrix.sum(0)
    matrix_sum = row_sums.sum()

    res = np.zeros_like(input_matrix, float)
    for i in range(num_rows):
        for j in range(num_cols):
            if input_matrix[i, j] == 0:
                res[i, j] = 0
            elif (row_sums[i] == 0) or (col_sums[j] == 0):
                res[i, j] = 0
            else:
                top = input_matrix[i, j] / matrix_sum
                bottom = (row_sums[i] / matrix_sum) * (col_sums[j] / matrix_sum)
                div = top/bottom
                if div <= 1:
                    res[i, j] = 0
                else:
                    res[i, j] = np.log(div)
    return res


def row_log_entropy(input_matrix: np.array,
                    ) -> np.array:
    print('Normalizing matrix by log entropy')

    num_rows = input_matrix.shape[0]
    res = np.zeros_like(input_matrix, float)

    # row probabilities
    row_prob_matrix = np.zeros_like(input_matrix, float)
    for i in range(num_rows):
        row_sum = input_matrix[i, :].sum()
        if row_sum == 0:
            print(f'Warning: Row {i} had sum of zero. Setting prob to 0')
        else:
            row_prob_matrix[i, :] = input_matrix[i, :] / row_sum

    log_frequencies = np.log(input_matrix + 1)
    for i in range(num_rows):
        # entropy = sigma[p(x) * log(1/p(x))]
        row_entropy = np.dot(row_prob_matrix[i, :],
                             np.log(1 / (1 + row_prob_matrix[i, :]))
                             )
        res[i, :] = log_frequencies[i, :] * row_entropy

    return res

# ////////////////////////////////////////////////// reductions


def reduce(input_matrix: np.array,
           reduce_type: str,
           reduce_size: int,
           ) -> np.array:
    if reduce_type == 'svd':
        reduced_matrix = reduce_svd(input_matrix, reduce_size)
    elif reduce_type == 'rva':
        reduced_matrix = reduce_rva(input_matrix, reduce_size)
    elif reduce_type is None:
        reduced_matrix = input_matrix
    else:
        raise AttributeError(f"Improper matrix reduction type '{reduce_type}'. "
                             "Must be 'svd', 'rva', or None")
    return reduced_matrix


def reduce_svd(input_matrix: np.array,
               reduce_size: int,
               ) -> np.array:
    print('Reducing matrix using SVD to {} singular values'.format(reduce_size))
    u, s, v = np.linalg.svd(input_matrix)
    # sparse_cooc_mat = sparse.csr_matrix(input_matrix).asfptype()
    # u, s, v = slinalg.svds(cooc_mat, k=reduce_size)

    reduced_matrix = u[:, 0:reduce_size]
    return reduced_matrix


def reduce_rva(input_matrix: np.array,
               reduce_size,
               mean: float = 0.0,
               std_dev: float = 1.0,
               ) -> np.array:
    print('Reducing matrix using RVA')
    num_rows = input_matrix.shape[0]
    random_vectors = np.random.normal(mean, std_dev, [num_rows, reduce_size])
    rva_matrix = np.zeros([num_rows, reduce_size], float)
    pbar = pyprind.ProgBar(num_rows, stream=sys.stdout)
    for i in range(num_rows):
        for j in range(num_rows):
            rva_matrix[i, :] += (input_matrix[i, j]*random_vectors[j, :])
        pbar.update()

    return rva_matrix
