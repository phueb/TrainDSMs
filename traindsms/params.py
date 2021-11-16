from dataclasses import dataclass, fields
from typing import Tuple, Optional


# submit jobs for one dsm at a time
DSM_NAME = ['count',     # 0
            'rnn',       # 1
            'glove',     # 2
            'w2v',       # 3
            ][1]

param2requests = {
    'num_epochs': [2, 4, 6, 8]
}

if DSM_NAME == 'count':
    param2default_dsm = {
        # window size means how many neighbors are considered in forward direction
        'count_type': ('ww', 'summed', 4, 'flat'),  # currently, sentence-boundary is respected automatically
        'norm_type': None,
        'reduce_type': (None, None),
    }

elif DSM_NAME == 'random':
    param2default_dsm = {
        'embed_size': 8,
        'random_type': 'normal',
    }

elif DSM_NAME == 'w2v':
    param2default_dsm = {
        'w2vec_type': 'sg',  # or 'cbow'
        'embed_size': 8,
        'window_size': 4,
        'num_epochs': 20,
    }

elif DSM_NAME == 'glove':
    param2default_dsm = {
        'embed_size': 8,
        'window_size': 7,
        'num_epochs': 20,
        'lr': 0.05,
    }

elif DSM_NAME == 'rnn':
    param2default_dsm = {
        'rnn_type': 'srn',
        'embed_size': 8,
        'train_percent': 1.0,
        'shuffle_per_epoch': True,
        'embed_init_range': 0.1,
        'dropout_prob': 0,
        'num_layers': 1,
        'seq_len': 4,
        'batch_size': 2,
        'num_epochs': 10,
        'learning_rate': (0.1, 0.95, 1),  # initial, decay, num_epochs_without_decay
        'grad_clip': None,
    }

else:
    raise NotImplementedError

param2requests['dsm'] = [DSM_NAME]

param2default = {
    # corpus
    'include_location': False,
    'include_location_specific_agents': False,
    'seed': 0,
    'num_epochs': 1,

    'dsm': None,

    'composition_fn': 'multiplication',  # todo when "native" the function should be native to the model, e.g. next-word predic5tion with rnn

}

param2default.update(param2default_dsm)


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


@dataclass
class RNNParams:
    rnn_type: str
    embed_size: int
    train_percent: float
    shuffle_per_epoch: bool
    embed_init_range: float
    dropout_prob: float
    num_layers: int
    seq_len: int
    batch_size: int
    num_epochs: int
    learning_rate: Tuple[float, float, float]
    grad_clip: Optional[float]

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})


@dataclass
class Word2VecParams:
    w2vec_type: str
    embed_size: int
    window_size: int
    num_epochs: int

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})


@dataclass
class GloveParams:
    embed_size: int
    window_size: int
    num_epochs: int
    lr: float

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})


@dataclass
class RandomControlParams:
    embed_size: int
    random_type: str

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})


@dataclass
class CTNParams:
    pass


@dataclass
class LONParams:
    pass


@dataclass
class Params:

    corpus_params: CorpusParams
    dsm: str
    dsm_params: CountParams
    composition_fn: str

    @classmethod
    def from_param2val(cls, param2val):

        # exclude keys from param2val which are added by Ludwig.
        # they are relevant to job submission only.
        tmp = {k: v for k, v in param2val.items()
               if k not in ['job_name', 'param_name', 'save_path', 'project_path']}

        if param2val['dsm'] == 'count':
            dsm_params = CountParams.from_param2val(tmp)
        elif param2val['dsm'] == 'random':
            dsm_params = CountParams.from_param2val(tmp)
        elif param2val['dsm'] == 'w2v':
            dsm_params = Word2VecParams.from_param2val(tmp)
        elif param2val['dsm'] == 'glove':
            dsm_params = GloveParams.from_param2val(tmp)
        elif param2val['dsm'] == 'rnn':
            dsm_params = RNNParams.from_param2val(tmp)
        else:
            raise AttributeError('Invalid arg to "dsm".')

        corpus_params = CorpusParams.from_param2val(tmp)
        return cls(dsm=param2val['dsm'],
                   composition_fn=param2val['composition_fn'],
                   corpus_params=corpus_params,
                   dsm_params=dsm_params,
                   )
