








class CountParams:
    count_type = [
        ['ww', 'concatenated',  7,  'linear'],
        ['wd', None, None, None],
    ]
    # norm_type = [None, 'row_sum', 'row_logentropy', 'tf_idf', 'ppmi']
    norm_type = ['ppmi']
    reduce_type = [
        # ['svd', 30],
        ['svd', 200],
        # ['svd', 500],
        # [None, None]  # TODO this makes expert training last too long
    ]


class RNNParams:
    rnn_type = ['srn', 'lstm']
    embed_size = [200]
    train_percent = [0.9]
    num_eval_steps = [1000]
    shuffle_per_epoch = [True]
    embed_init_range = [0.1]
    dropout_prob = [0]
    num_layers = [1]
    num_steps = [7]
    batch_size = [64]
    num_epochs = [20]  # 20 is only slightly better than 10
    learning_rate = [[0.01, 1.0, 10]]  # initial, decay, num_epochs_without_decay
    grad_clip = [None]


class Word2VecParams:
    w2vec_type = ['sg', 'cbow']
    embed_size = [200]
    window_size = [7]
    num_epochs = [20]


class GloveParams:
    glove_type = []  # TODO
    embed_size = [200]
    window_size = [7]
    num_epochs = [20]  # semantic novice ba:  10: 0.64, 20: 0.66,  40: 0.66
    lr = [0.05]


class RandomControlParams:
    embed_size = [200]
    random_type = ['normal']


# TODO
# create all possible hyperparameter configurations
update_d = {'corpus_name': config.Corpus.name, 'num_vocab': config.Corpus.num_vocab}
param2val_list = list_all_param2vals(RandomControlParams, update_d) + \
                 list_all_param2vals(CountParams, update_d) + \
                 list_all_param2vals(RNNParams, update_d) + \
                 list_all_param2vals(Word2VecParams, update_d)