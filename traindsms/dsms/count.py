import numpy as np
from cytoolz import itertoolz
import pyprind
import sys
import time
from scipy.sparse import linalg as slinalg
from scipy import sparse
from typing import List

from traindsms.params import CountParams

PAD = '*PAD*'
VERBOSE = False


class CountDSM:
    def __init__(self,
                 params: CountParams,
                 numeric_docs: List[List[int]],
                 ):
        self.params = params

        self.numeric_docs = numeric_docs
        self.vocab_size = np.array(self.numeric_docs).max().item() + 1

    # ////////////////////////////////////////////////// word-by-word

    def create_ww_matrix_fast(self):  # no function call overhead - twice as fast
        window_type = self.params.count_type[1]
        window_size = self.params.count_type[2]
        window_weight = self.params.count_type[3]

        # count
        num_docs = len(self.numeric_docs)
        pbar = pyprind.ProgBar(num_docs, stream=sys.stdout)
        count_matrix = np.zeros([self.vocab_size, self.vocab_size], int)
        print('\nCounting word-word co-occurrences in {}-word moving window'.format(window_size))
        for token_ids in self.numeric_docs:
            token_ids += [PAD] * window_size  # add padding such that all co-occurrences in last window are captured
            if VERBOSE:
                print(token_ids)
            windows = itertoolz.sliding_window(window_size + 1, token_ids)  # + 1 because window consists of t2s only
            for w in windows:
                for t1_id, t2_id, dist in zip([w[0]] * window_size,
                                              w[1:],
                                              range(window_size)):
                    # increment
                    if t1_id == PAD or t2_id == PAD:
                        continue
                    if window_weight == "linear":
                        print(count_matrix.shape)
                        print(t1_id)
                        print(t2_id)
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
        # count
        num_docs = len(self.numeric_docs)
        pbar = pyprind.ProgBar(num_docs, stream=sys.stdout)
        count_matrix = np.zeros([self.vocab_size, num_docs], int)
        print('\nCounting word occurrences in {} documents'.format(num_docs))
        for i in range(num_docs):
            for j in self.numeric_docs[i]:
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
        print('Completed count in {}'.format(time.time() - start))

        # normalize + reduce
        norm_matrix, dimensions = self.normalize(count_matrix, self.params.norm_type)
        reduced_matrix, dimensions = self.reduce(norm_matrix, self.params.reduce_type[0], self.params.reduce_type[1])

        return reduced_matrix  # for unittest

    # ////////////////////////////////////////////////// normalizations

    def normalize(self, input_matrix, norm_type):
        if norm_type == 'row_sum':
            norm_matrix, dimensions = self.norm_rowsum(input_matrix)
        elif norm_type == 'col_sum':
            norm_matrix, dimensions = self.norm_colsum(input_matrix)
        elif norm_type == 'tf_idf':
            norm_matrix, dimensions = self.norm_tfidf(input_matrix)
        elif norm_type == 'row_logentropy':
            norm_matrix, dimensions = self.row_logentropy(input_matrix)
        elif norm_type == 'ppmi':
            norm_matrix, dimensions = self.norm_ppmi(input_matrix)
        elif norm_type is None:
            norm_matrix = input_matrix
            dimensions = input_matrix.shape[1]
        else:
            raise AttributeError("Improper matrix normalization type '{}'. "
                                 "Must be 'row_sum', 'col_sum', 'row_logentropy', 'tf-idf', 'ppmi', or 'none'".format(norm_type))
        return norm_matrix, dimensions

    def norm_rowsum(self, input_matrix):
        print('\nNormalizing matrix by row sums...')
        num_rows = len(input_matrix[:,0])
        num_cols = len(input_matrix[0,:])
        output_matrix = np.zeros([num_rows, num_cols], float)
        pbar = pyprind.ProgBar(num_rows, stream=sys.stdout)
        for i in range(num_rows):
            if input_matrix[i,:].sum() == 0:
                print('    Warning: Row {} had sum of zero. Setting prob to 0'.format(i))
            else:
                output_matrix[i,:] = input_matrix[i,:] / input_matrix[i,:].sum()
            pbar.update()
        return output_matrix, num_cols

    def norm_colsum(self, input_matrix):
        print('\nNormalizing matrix by column sums...')
        num_rows = len(input_matrix[:,0])
        num_cols = len(input_matrix[0,:])
        output_matrix = np.zeros([num_rows, num_cols], float)
        pbar = pyprind.ProgBar(num_cols, stream=sys.stdout)
        for i in range(num_cols):
            if input_matrix[:,i].sum() == 0:
                print('    Warning: Column {} had sum of zero. Setting prob to 0'.format(i))
            else:
                output_matrix[:,i] = input_matrix[:,i] / input_matrix[:,i].sum()
            pbar.update()
        return output_matrix, num_cols

    def norm_tfidf(self, input_matrix):
        print('\nNormalizing matrix by tf-idf...')
        num_rows = len(input_matrix[:,0])
        num_cols = len(input_matrix[0,:])
        print('Calculating column probs')
        pbar = pyprind.ProgBar(num_cols, stream=sys.stdout)
        colprob_matrix = np.zeros([num_rows, num_cols], float)
        for i in range(num_cols):
            if input_matrix[:,i].sum() == 0:
                print('    Warning: Column {} had sum of zero. Setting prob to 0'.format(i))
            else:
                colprob_matrix[:,i] = input_matrix[:,i] / input_matrix[:,i].sum()
            pbar.update()
        print('Calculating tf-idf scores')
        output_matrix = np.zeros([num_rows, num_cols], float)
        pbar = pyprind.ProgBar(num_rows, stream=sys.stdout)
        for i in range(num_rows):
            col_occ_count = np.count_nonzero(input_matrix[i,:]) + 1
            row_idf = float(num_cols) / col_occ_count
            for j in range(num_cols):
                output_matrix[i,j] = colprob_matrix[i,j] / row_idf
            pbar.update()
        return output_matrix, num_cols

    def norm_ppmi(self, input_matrix):
        print('\nNormalizing matrix by ppmi')
        num_rows = len(input_matrix[:,0])
        num_cols = len(input_matrix[0,:])

        row_sums = input_matrix.sum(1)
        col_sums = input_matrix.sum(0)
        matrix_sum = row_sums.sum()

        output_matrix = np.zeros([num_rows, num_cols], float)
        pbar = pyprind.ProgBar(num_rows, stream=sys.stdout)
        for i in range(num_rows):
            for j in range(num_cols):
                if input_matrix[i, j] == 0:
                    output_matrix[i, j] = 0
                elif (row_sums[i] == 0) or (col_sums[j] == 0):
                    output_matrix[i, j] = 0
                else:
                    top = input_matrix[i, j] / matrix_sum
                    bottom = (row_sums[i] / matrix_sum) * (col_sums[j] / matrix_sum)
                    div = top/bottom
                    if div <= 1:
                        output_matrix[i, j] = 0
                    else:
                        output_matrix[i, j] = np.log(div)
            pbar.update()
        return output_matrix, num_cols

    def row_logentropy(self, input_matrix):
        print('\nNormalizing matrix by log entropy')
        num_rows = len(input_matrix[:,0])
        num_cols = len(input_matrix[0,:])
        output_matrix = np.zeros([num_rows, num_cols], float)

        print('Computing row probabilities')
        row_prob_matrix = np.zeros([num_rows, num_cols], float)
        pbar = pyprind.ProgBar(num_rows, stream=sys.stdout)
        for i in range(num_rows):
            if input_matrix[i,:].sum() == 0:
                print('    Warning: Row {} had sum of zero. Setting prob to 0'.format(i))
            else:
                row_prob_matrix[i,:] = input_matrix[i,:] / input_matrix[i,:].sum()
            pbar.update()

        print('Computing entropy scores')
        log_freqs = np.log(input_matrix + 1)
        pbar = pyprind.ProgBar(num_rows, stream=sys.stdout)
        for i in range(num_rows):
            row_entropy = np.dot(row_prob_matrix[i, :], np.log(row_prob_matrix[i, :] + 1))
            global_weight = 1 + (row_entropy / np.log(num_cols + 1))

            for j in range(num_cols):
                output_matrix[i, j] = log_freqs[i, j] * global_weight
            pbar.update()

        return output_matrix, num_cols

    # ////////////////////////////////////////////////// reductions

    def reduce(self, input_matrix, reduce_type, reduce_size):
        if reduce_type == 'svd':
            reduced_matrix, dimensions = self.reduce_svd(input_matrix, reduce_size)
        elif reduce_type == 'rva':
            reduced_matrix, dimensions = self.reduce_rva(input_matrix, reduce_size)
        elif reduce_type is None:
            reduced_matrix = input_matrix
            dimensions = input_matrix.shape[1]
        else:
            raise AttributeError("Improper matrix reduction type '{}'. "
                                 "Must be 'svd', 'rva', or 'none'".format(reduce_type))
        return reduced_matrix, dimensions

    def reduce_svd(self, input_matrix, dimensions):
        print('\nReducing matrix using SVD to {} singular values'.format(dimensions))
        # u, s, v = np.linalg.svd(input_matrix)
        sparse_cooc_mat = sparse.csr_matrix(input_matrix).asfptype()
        u, s, v = slinalg.svds(sparse_cooc_mat, k=dimensions)

        reduced_matrix = u[:, 0:dimensions]
        return reduced_matrix, dimensions

    def reduce_rva(self, input_matrix, length, mean=0, stdev=1):
        print('\nReducing matrix using RVA')
        num_rows = len(input_matrix[:, 0])
        num_cols = len(input_matrix[0, :])
        random_vectors = np.random.normal(mean,stdev,[num_rows, length])
        rva_matrix = np.zeros([num_rows, length], float)
        pbar = pyprind.ProgBar(num_rows, stream=sys.stdout)
        for i in range(num_rows):
            for j in range(num_rows):
                rva_matrix[i,:] += (input_matrix[i,j]*random_vectors[j,:])
            pbar.update()

        return rva_matrix, length