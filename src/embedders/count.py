import numpy as np
from cytoolz import itertoolz
import pyprind
import sys

from src.utils import matrix_to_w2e
from src.embedders.base import EmbedderBase
from src.params import CountParams
from src import config

PAD = '*PAD*'
VERBOSE = False


class CountEmbedder(EmbedderBase):
    def __init__(self, param2ids, param2val):
        super().__init__(param2val)
        self.count_type = CountParams.count_type[param2ids.count_type]
        self.norm_type = CountParams.norm_type[param2ids.norm_type]
        self.reduce_type = CountParams.reduce_type[param2ids.reduce_type]
        #
        self.name = self.count_type[0]

    # ////////////////////////////////////////////////// word-by-word

    def create_ww_matrix(self):
        window_type = self.count_type[1]
        window_size = self.count_type[2]
        window_weight = self.count_type[3]

        def increment(mat, t1_id, t2_id, dist):
            if t1_id == PAD or t2_id == PAD:
                return
            if window_weight == "linear":
                mat[t1_id, t2_id] += window_size - dist
            elif window_weight == "flat":
                mat[t1_id, t2_id] += 1
            if VERBOSE:
                print('Incrementing @ row {:>3} col {:>3}'.format(t1_id, t2_id))

        def update_matrix(mat, ids):
            ids += [PAD] * window_size  # add padding such that all co-occurrences in last window are captured
            if VERBOSE:
                print(ids)
            windows = itertoolz.sliding_window(window_size, ids)
            for w in windows:
                for t1_id, t2_id, dist in zip([w[0]] * (window_size - 1),
                                              w[1:],
                                              range(1, window_size)):
                    increment(mat, t1_id, t2_id, dist)
            if VERBOSE:
                print()

        # count
        num_docs = len(self.numeric_docs)
        pbar = pyprind.ProgBar(num_docs)
        count_matrix = np.zeros([config.Corpus.num_vocab, config.Corpus.num_vocab], int)
        print('\nCounting word-word co-occurrences in {}-word moving window'.format(window_size))
        for token_ids in self.numeric_docs:
            update_matrix(count_matrix, token_ids)
            pbar.update()

        # window_type
        if window_type == 'forward':
            final_matrix = count_matrix
        elif window_type == 'backward':
            final_matrix = count_matrix.transpose()
        elif window_type == 'summed':
            final_matrix = count_matrix + count_matrix.transpose()
        elif window_type == 'concatenate':
            final_matrix = np.concatenate((count_matrix, count_matrix.transpose()))
        else:
            raise AttributeError('Invalid arg to "window_type".')
        return final_matrix

    # ////////////////////////////////////////////////// word-by-document

    def create_wd_matrix(self):
        # count
        num_docs = len(self.numeric_docs)
        pbar = pyprind.ProgBar(num_docs)
        count_matrix = np.zeros([config.Corpus.num_vocab, num_docs], int)
        print('\nCounting word occurrences in {} documents'.format(num_docs))
        for i in range(num_docs):
            for j in self.numeric_docs[i]:
                count_matrix[j, i] += 1
            pbar.update()
        return count_matrix

    # ////////////////////////////////////////////////// train

    def train(self):
        # count
        if self.count_type[0] == 'ww':
            count_matrix = self.create_ww_matrix()
        elif self.count_type[0] == 'wd':
            count_matrix = self.create_wd_matrix()
        else:
            raise AttributeError('Invalid arg to "count_type".')
        # normalize + reduce
        norm_matrix, dimensions = self.normalize(count_matrix, self.norm_type)
        reduced_matrix, dimensions = self.reduce(norm_matrix, self.reduce_type[0], self.reduce_type[1])
        # to w2e
        w2e = matrix_to_w2e(reduced_matrix, self.vocab)
        return w2e

    # ////////////////////////////////////////////////// normalizations

    def normalize(self, input_matrix, norm_type):
        if norm_type == 'row_sum':
            norm_matrix, dimensions = self.norm_rowsum(input_matrix)
        elif norm_type == 'col_sum':
            norm_matrix, dimensions = self.norm_rowsum(input_matrix)
        elif norm_type == 'tf_idf':
            norm_matrix, dimensions = self.norm_rowsum(input_matrix)
        elif norm_type == 'row_logentropy':
            norm_matrix, dimensions = self.norm_rowsum(input_matrix)
        elif norm_type == 'ppmi':
            norm_matrix, dimensions = self.norm_rowsum(input_matrix)
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
                print('    Warning: Row {} ({}) had sum of zero. Setting prob to 0'.format(i, self.vocab[i]))
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
                print('    Warning: Column {} had sum of zero. Setting prob to 0'.format(i, self.vocab[i]))
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
                print('    Warning: Column {} had sum of zero. Setting prob to 0'.format(i, self.vocab[i]))
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
                print('    Warning: Row {} ({}) had sum of zero. Setting prob to 0'.format(i, self.vocab[i]))
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
        elif reduce_type == 'none':
            reduced_matrix = input_matrix
            dimensions = input_matrix.shape[1]
        else:
            raise AttributeError("Improper matrix reduction type '{}'. "
                                 "Must be 'svd', 'rva', or 'none'".format(reduce_type))
        return reduced_matrix, dimensions

    def reduce_svd(self, input_matrix, dimensions):
        print('\nReducing matrix using SVD to {} singular values'.format(dimensions))
        u, s, v = np.linalg.svd(input_matrix)
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