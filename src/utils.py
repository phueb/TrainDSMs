from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bayes_opt import BayesianOptimization
import spacy

from src import config

nlp = spacy.load('en_core_web_sm')


def calc_balanced_accuracy(calc_signals, sims_mean, verbose=True):

    def calc_probes_fs(thr):
        tp, tn, fp, fn = calc_signals(thr)
        precision = np.divide(tp + 1e-10, (tp + fp + 1e-10))
        sensitivity = np.divide(tp + 1e-10, (tp + fn + 1e-10))  # aka recall
        fs = 2 * (precision * sensitivity) / (precision + sensitivity)
        return fs

    def calc_probes_ba(thr):
        tp, tn, fp, fn = calc_signals(thr)
        specificity = np.divide(tn + 1e-10, (tn + fp + 1e-10))
        sensitivity = np.divide(tp + 1e-10, (tp + fn + 1e-10))  # aka recall
        ba = (sensitivity + specificity) / 2  # balanced accuracy
        return ba

    # make thr range
    thr1 = max(0.0, round(min(0.9, round(sims_mean, 2)) - 0.1, 2))  # don't change
    thr2 = round(thr1 + 0.2, 2)
    # use bayes optimization to find best_thr
    if verbose:
        print('Finding best thresholds between {} and {} using bayesian-optimization...'.format(thr1, thr2))
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
    if config.Task.metric == 'fs':
        fun = calc_probes_fs
    elif config.Task.metric == 'ba':
        fun = calc_probes_ba
    else:
        raise AttributeError('rnnlab: Invalid arg to "metric".')
    bo = BayesianOptimization(fun, {'thr': (thr1, thr2)}, verbose=verbose)
    bo.explore({'thr': [sims_mean]})
    bo.maximize(init_points=2, n_iter=config.Task.num_opt_steps,
                acq="poi", xi=0.001, **gp_params)  # smaller xi: exploitation
    best_thr = bo.res['max']['max_params']['thr']
    # use best_thr
    results = fun(best_thr)
    res = np.mean(results)
    return res


def w2e_to_sims(w2e, row_words, col_words, method):
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


