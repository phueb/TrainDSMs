from dataclasses import dataclass, fields
from typing import Tuple, Optional

param2requests = {
    'dsm': ['count'],
}


param2default = {
    # corpus
    'include_location': False,
    'include_location_specific_agents': False,
    'seed': 0,
    'num_epochs': 1,

    'dsm': None,

    # count
    'count_type': ('ww', 'concatenated',  4,  'linear'),  # Todo how to respect sequence boundary?
    'norm_type': None,
    'reduce_type': (None, None),

    # random
    'embed_size': 200,
    'random_type': 'normal',

    # gloVe
    # 'embed_size': 200,  # TODO how to specify multiple params with the same name?
    # 'window_size': 7,
    # 'num_epochs': 20,
    # 'lr': 0.05,

}


@dataclass
class CorpusParams:
    include_location: bool
    include_location_specific_agents: bool
    seed: int
    num_epochs: int

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})


@dataclass
class CountParams:
    count_type: Tuple[str, Optional[str], Optional[int], Optional[str]]
    # ('ww', 'concatenated',  4,  'linear')
    # ('wd', None, None, None)
    norm_type: Optional[str]  # e.g. None, 'row_sum', 'row_logentropy', 'tf_idf', 'ppmi'
    reduce_type: Tuple[Optional[str], Optional[int]]  # e.g. ('svd', 200) or (None, None)

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})


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
    embed_size: int
    window_size: int
    num_epochs: int
    lr: float


class RandomControlParams:
    embed_size: int
    random_type: str


@dataclass
class Params:

    corpus_params: CorpusParams
    dsm: str
    count_params: CountParams

    @classmethod
    def from_param2val(cls, param2val):
        """
        instantiate class.
        exclude keys from param2val which are added by Ludwig.
        they are relevant to job submission only.
        """
        tmp = {k: v for k, v in param2val.items()
               if k not in ['job_name', 'param_name', 'save_path', 'project_path']}
        return cls(dsm=param2val['dsm'],
                   corpus_params=CorpusParams.from_param2val(tmp),
                   count_params=CountParams.from_param2val(tmp),
                   )