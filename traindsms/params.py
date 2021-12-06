"""
This is where a user specifies which model to train on which data
"""

from dataclasses import dataclass, fields
from typing import Tuple, Optional, List


# submit jobs for one dsm at a time
DSM_NAME = ['count',        # 0
            'rnn',          # 1
            'transformer',  # 2
            'glove',        # 3
            'w2v',          # 4
            'lon',          # 5
            'ctn',          # 6
            ][1]

param2requests = {
    'seed': [1],
    'composition_fn': ['native'],

    'num_epochs': [3000000]

    # TODO lstm?


}

# ################################################## End of user-editable settings

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
        # architecture
        'rnn_type': 'srn',
        'embed_size': 8,            # 8 is best
        'num_layers': 1,
        'train_percent': 1.0,
        # optimization
        'embed_init_range': 0.1,    # 0.1 is good
        'dropout_prob': 0.0,        # must be 0.0 with num_layers=1
        'batch_size': 64,           # 64 is good
        'num_epochs': 10,           # no more than 10 needed with batch_size=64
        'learning_rate': 0.1,       # 0.1 with batch_size=64
        'grad_clip': 1.0,
        'lr_decay': 0.001,          # 0.001 but no larger
        'weight_decay': 0.0,        # keep at 0
    }


elif DSM_NAME == 'transformer':
    param2default_dsm = {
        # architecture
        'transformer_type': 'gpt2',
        'embed_size': 8,                # no larger than 8
        'inner_size': 4,                # must be divisible by embed_size
        'resid_pdrop': 0.0,             # 0 is best with lr=0.09
        'num_layers': 1,                # 1 layer is better than multiple
        'num_heads': 1,                 # 1 is best
        'seq_len': 8,                   # this must be larger than the largest sentence in the corpus
        # optimization
        'batch_size': 128,              # should be smaller than 576 (size of complete block)
        'num_epochs': 20,               # 20 is best with num_blocks=400, less or more hurts exp2b accuracy
        'learning_rate': 0.01,          # is decayed during training with Adam optimizer
        'weight_decay': 0.0,            # 0.0 is best
        'adam_beta2': 0.999,            # default, robust to small changes
        'adam_epsilon': 1e-08,          # default, robust to small changes
        'label_smoothing': 0.0,         # default, robust to small changes
        'initializer_range': 0.003,     # 0.003 is better than default 0.002

    }

elif DSM_NAME == 'lon':
    param2default_dsm = {

        # TODO the LON is currently built from corpus directly rather than co-mat

        # todo: the co-occurrence matrix will be it's adjacency matrix, and the spreading activation functions on the adjacency matrix

        'count_type': ('ww', 'summed', 4, 'flat'),  # currently, sentence-boundary is respected automatically
        'norm_type': None,
        'excluded_tokens': None,
    }

elif DSM_NAME == 'ctn':
    param2default_dsm = {
        'excluded_tokens': None,
    }

else:
    raise NotImplementedError

param2requests['dsm'] = [DSM_NAME]

param2default = {
    'dsm': None,
    'composition_fn': 'multiplication',
}

param2default_corpus = {
    'seed': 0,
    'complete_block': True,  # generate all possible samples, in addition to num_blocks, e.g. 576 for exp2
    'num_blocks': 400,  # 400 produces better loss in transformer than fewer blocks
    'include_location': False,
    'include_location_specific_agents': False,
}

# avoid keys with the same name
for k in param2default_corpus:
    assert k not in param2default_dsm
param2default.update(param2default_corpus)
param2default.update(param2default_dsm)

param2debug = {
    'complete_block': True,
    'num_blocks': 0,
    'dsm': 'rnn',
    'num_epochs': 10,
}


if 'dropout_prob' in param2requests:
    for dp in param2requests['dropout_prob']:
        if dp > 0.0:
            if 'num_layers' in param2requests:
                if param2requests['num_layers'] == 1:
                    raise ValueError('Cannot use non-zero dropout when num_layers=1')
            else:
                if param2default['num_layers'] == 1:
                    raise ValueError('Cannot use non-zero dropout when num_layers=1')

# ################################################## End of default-settings

@dataclass
class CorpusParams:
    include_location: bool
    include_location_specific_agents: bool
    seed: int
    num_blocks: int
    complete_block: bool

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
    # architecture
    rnn_type: str
    embed_size: int
    num_layers: int
    train_percent: float
    # optimization
    embed_init_range: float
    dropout_prob: float
    batch_size: int
    num_epochs: int
    learning_rate: float
    grad_clip: Optional[float]
    lr_decay: float
    weight_decay: float

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})


@dataclass
class TransformerParams:
    # architecture
    transformer_type: str
    embed_size: int
    inner_size: int
    resid_pdrop: float
    num_layers: int
    num_heads: int
    seq_len: int
    # optimization
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    adam_beta2: float
    adam_epsilon: float
    label_smoothing: float
    initializer_range: float

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
    excluded_tokens: Optional[Tuple[str]]

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})


@dataclass
class LONParams:
    excluded_tokens: Optional[Tuple[str]]

    count_type: Tuple[str, Optional[str], Optional[int], Optional[str]]
    norm_type: Optional[str]  # e.g. None, 'row_sum', 'row_logentropy', 'tf_idf', 'ppmi'

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})


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
        elif param2val['dsm'] == 'transformer':
            dsm_params = TransformerParams.from_param2val(tmp)
        elif param2val['dsm'] == 'lon':
            dsm_params = LONParams.from_param2val(tmp)
        elif param2val['dsm'] == 'ctn':
            dsm_params = CTNParams.from_param2val(tmp)
        else:
            raise AttributeError(f'Invalid arg to "dsm" "{param2val["dsm"]}".')

        corpus_params = CorpusParams.from_param2val(tmp)
        return cls(dsm=param2val['dsm'],
                   composition_fn=param2val['composition_fn'],
                   corpus_params=corpus_params,
                   dsm_params=dsm_params,
                   )
