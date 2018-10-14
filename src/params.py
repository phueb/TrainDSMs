import numpy as np
from itertools import cycle


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


def make_param2ids(paramsClass):
    """
    return list of mappings from param name to integer which is index to possible param values
    all possible combinations are returned
    """
    lengths = []
    params = sorted([(k, v) for k,v in paramsClass.__dict__.items()
                     if not k.startswith('_')])
    for k, v in params:
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
    # map param names to integers corresponding to which param value to use
    for ids in param_ids:
        d = {k: i for (k, v), i in zip(params, ids)}
        view = ObjectView(d)
        print('==========================================================================')
        for (k, v), i in zip(params, ids):
            print(k, v[i])
        yield view


class CountParams:
    count_type = [('wd', None, None, None),
                  ('ww', 'forward',  7,  'linear'),
                  ('ww', 'forward',  7,  'flat'),
                  ('ww', 'backward', 7,  'linear'),
                  ('ww', 'backward', 7,  'flat'),
                  ('ww', 'forward',  16, 'linear'),
                  ('ww', 'forward',  16, 'flat'),
                  ('ww', 'backward', 16, 'linear'),
                  ('ww', 'backward', 16, 'flat')]
    norm_type = [None, 'row_logentropy', 'tf_idf', 'ppmi']
    reduce_type = [('svd', 200), ('svd', 30), (None, None)]


class RNNParams:
    rnn_type = ['lstm', 'srn']
    embed_size = [512]
    train_percent = [0.9]
    num_eval_steps = [1000]
    shuffle_per_epoch = [True]
    embed_init_range = [0.1]
    dropout_prob = [0]
    num_layers = [1]
    num_steps = [7]
    batch_size = [64]
    num_epochs = [20]
    learning_rate = [(0.1, 0.85, 10)]  # initial, decay, num_epochs_without_decay
    grad_clip = [None]


class Word2VecParams:
    w2vec_type = ['cbow', 'sg']
    embed_size = [32, 128, 512]
    window_size = [7, 16]
    num_epochs = [20]


class RandomControlParams:
    embed_size = [512]
    distribution = ['normal', 'uniform']