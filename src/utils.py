from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sortedcontainers import SortedDict


def matrix_to_simmat(input_matrix, method):
    if method == 'cosine':
        res = cosine_similarity(input_matrix)
    else:
        raise NotImplemented  # TODO how to convert euclidian distance to sim measure?
    return res


def w2e_to_matrix(w2e, probes=None):
    if probes is None:
        probes = w2e.keys()
    embeds = []
    for word, embed in w2e.items():  # assumes sorted dict
        if word in probes:
            embeds.append(embed)
    res = np.vstack(embeds)
    print('Converted w2e to matrix with shape {}'.format(res.shape))
    return res


def matrix_to_w2e(input_matrix, vocab):
    res = SortedDict()
    for n, w in enumerate(vocab):
        res[w] = input_matrix[n]
    return res


def print_matrix(vocab, matrix, precision, row_list=None, column_list=None):

    t2id = {t: i for i, t in enumerate(vocab)}

    print()

    if row_list != None:
        for i in range(len(row_list)):
            if row_list[i] in t2id:
                row_index = t2id[row_list[i]]
                print('{:<15}   '.format(row_list[i]), end='')

                if column_list != None:
                    for j in range(len(column_list)):
                        if column_list[j] in t2id:
                            column_index = t2id[column_list[j]]
                        print('{val:6.{precision}f}'.format(precision=precision, val=matrix[row_index, column_index]), end='')
                    print()
                else:
                    for i in range(len(matrix[:, 0])):
                        print('{:<15}   '.format(vocab[i]), end='')
                        for j in range(len(matrix[i, :])):
                            print('{val:6.{precision}f}'.format(precision=precision, val=matrix[i, j]), end='')
                        print()
    else:
        for i in range(len(matrix[:, 0])):
            print('{:<15}   '.format(vocab[i]), end='')
            for j in range(len(matrix[i, :])):
                print('{val:6.{precision}f}'.format(precision=precision, val=matrix[i, j]), end='')
            print()