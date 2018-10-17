from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy

from src import config

nlp = spacy.load('en_core_web_sm')


def w2e_to_sims(w2e, row_words, col_words, method):  # TODO test
    x = np.vstack([w2e[w] for w in row_words])
    y = np.vstack([w2e[w] for w in col_words])
    # sim
    if method == 'cosine':
        res = cosine_similarity(x, y)
    else:
        raise NotImplemented  # TODO how to convert euclidian distance to sim measure?
    return np.around(res, config.Embeddings.precision)


def print_matrix(vocab, matrix, precision, row_list=None, column_list=None):
    w2id = {w: i for i, w in enumerate(vocab)}
    if row_list is not None:
        for i in range(len(row_list)):
            if row_list[i] in w2id:
                row_index = w2id[row_list[i]]
                print('{:<15}   '.format(row_list[i]), end='')
                if column_list is not None:
                    for j in range(len(column_list)):
                        if column_list[j] in w2id:
                            column_index = w2id[column_list[j]]
                        print('{val:6.{precision}f}'.format(precision=precision,
                                                            val=matrix[row_index, column_index]), end='')
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


