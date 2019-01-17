import numpy as np
from itertools import cycle, chain

from src import config


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


def to_embedder_name(param2val):
    if 'random_type' in param2val:
        return 'random_{}'.format(param2val['random_type'])
    elif 'rnn_type' in param2val:
        return param2val['rnn_type']
    elif 'w2vec_type' in param2val:
        return param2val['w2vec_type']
    elif 'count_type' in param2val:
        return param2val['count_type'][0]
    elif 'glove_type' in param2val:
        return param2val['glove_type']
    else:
        raise RuntimeError('Unknown embedder name')


def iter_over_cycles(d):
    # lengths
    param2opts = sorted([(k, v) for k, v in d.items()
                         if not k.startswith('_')])
    lengths = []
    for k, v in param2opts:
        lengths.append(len(v))
    total = np.prod(lengths)
    num_lengths = len(lengths)
    # cycles
    cycles = []
    prev_interval = 1
    for n in range(num_lengths):
        l = np.concatenate([[i] * prev_interval for i in range(lengths[n])])
        if n != num_lengths - 1:
            c = cycle(l)
        else:
            c = l
        cycles.append(c)
        prev_interval *= lengths[n]
    # iterate over cycles, in effect retrieving all combinations
    param_ids = []
    for n, i in enumerate(zip(*cycles)):
        param_ids.append(i)
    assert sorted(list(set(param_ids))) == sorted(param_ids)
    assert len(param_ids) == total
    return param2opts, param_ids


def make_param2val_list(params_class1, params_class2):
    # merge
    meta_d = {'corpus_name': [config.Corpus.name], 'num_vocab': [config.Corpus.num_vocab]}
    merged_d = {k: v for k, v in chain(params_class1.__dict__.items(),
                                       params_class2.__dict__.items(),
                                       meta_d.items())}
    #
    param2opts, param_ids = iter_over_cycles(merged_d)
    #
    res = []
    for ids in param_ids:
        param2val = {k: v[i] for (k, v), i in zip(param2opts, ids)}
        res.append(param2val)
    return res


def gen_combinations(params_class):
    """
    return list of mappings from param name to integer which is index to possible param values
    all possible combinations are returned
    """
    d = params_class.__dict__
    #
    param2opts, param_ids = iter_over_cycles(d)
    # map param names to integers corresponding to which param value to use
    for ids in param_ids:
        param2val = {k: v[i] for (k, v), i in zip(param2opts, ids)}
        param2val.update({'corpus_name': config.Corpus.name, 'num_vocab': config.Corpus.num_vocab})
        yield param2val


class CountParams:
    count_type = [
        ['wd', None, None, None],
        ['ww', 'concatenated',  7,  'linear'],
        ['ww', 'concatenated', 16,  'linear'],
    ]
    # norm_type = [None, 'row_logentropy', 'tf_idf', 'ppmi']
    norm_type = ['row_sum']
    reduce_type = [
        ['svd', 30],
        ['svd', 200],
        ['svd', 500],
        [None, None]
    ]


class RNNParams:
    rnn_type = ['srn', 'lstm']
    embed_size = [30, 200, 500]
    train_percent = [0.9]
    num_eval_steps = [1000]
    shuffle_per_epoch = [True]
    embed_init_range = [0.1]
    dropout_prob = [0]
    num_layers = [1]
    num_steps = [7, 16]
    batch_size = [64]
    num_epochs = [20]  # 20 is only slightly better than 10
    learning_rate = [[0.01, 1.0, 10]]  # initial, decay, num_epochs_without_decay
    grad_clip = [None]


class Word2VecParams:
    w2vec_type = ['sg', 'cbow']
    embed_size = [30, 200, 500]
    window_size = [7, 16]
    num_epochs = [20]


class GloveParams:
    glove_type = []  # TODO
    embed_size = [30, 200, 500]
    window_size = [7, 16]
    num_epochs = [20]  # semantic novice ba:  10: 0.64, 20: 0.66,  40: 0.66
    lr = [0.05]


class RandomControlParams:
    embed_size = [30, 200, 500]
    random_type = ['normal']