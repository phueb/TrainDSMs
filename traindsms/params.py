from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CountParams:
    count_type: Tuple[str, Optional[str], Optional[int], Optional[str]]
    # ('ww', 'concatenated',  7,  'linear')
    # ('wd', None, None, None)
    norm_type: Optional[str]  # e.g. None, 'row_sum', 'row_logentropy', 'tf_idf', 'ppmi'
    reduce_type: Tuple[Optional[str], Optional[int]]  # e.g. ('svd', 200) or (None, None)

    @classmethod
    def from_param2val(cls, param2val):
        """
        instantiate class.
        exclude keys from param2val which are added by Ludwig.
        they are relevant to job submission only.
        """
        kwargs = {k: v for k, v in param2val.items()
                  if k not in ['job_name', 'param_name', 'save_path', 'project_path']}
        return cls(**kwargs)


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

