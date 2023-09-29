"""
This is where a user specifies which model to train on which data
"""

from dataclasses import dataclass, fields
from typing import Tuple, Optional, Union


# submit jobs for one dsm at a time
DSM_NAME = ['count',        # 0
            'lon',          # 1
            'rnn',          # 2
            'transformer',  # 3
            'w2v',          # 4
            'ctn',          # 5
            'random',       # 6
            ][0]

param2requests = {

    # count
    'add_with': [False],
    'composition_fn': ['componential'],
    'reduce_type': [('svd', 20), ('svd', 22), ('svd', 24), ('svd', 26), ('svd', 28), ('svd', 30), ('svd', 32), ('svd', 34), ('svd', 36),],

    # lon
    # 'add_with': [True, False],
    # 'context_size': [1, 2],

    # srn
    # 'add_with': [True],
    # 'rnn_type': ['srn'],
    # 'composition_fn': ['native'],
    # 'add_reversed_seq': [True, False],
    # 'strict_compositional': [False],
    # 'omit_type_2_verb_and_exp_theme': [False],
    # 'num_epochs': [4],  # no lower than 4
    # 'learning_rate': [0.06],  # no lower than 0.05
    # 'embed_init_range': [0.1],
    # 'num_layers': [2],

    # lstm
    # 'add_with': [True],
    # 'rnn_type': ['lstm'],
    # 'composition_fn': ['native'],
    # 'strict_compositional': [False, True],
    # 'omit_type_2_verb_and_exp_theme': [False, True],
    # 'add_reversed_seq': [True],
    # 'num_epochs': [4],  # 4 is best
    # 'learning_rate': [0.06],  # no lower than 0.05
    # 'embed_init_range': [0.05],  # 0.05 is best
    # 'num_layers': [2],

    # transformer
    # 'add_with': [True],
    # 'strict_compositional': [False],
    # 'omit_type_2_verb_and_exp_theme': [False],
    # 'composition_fn': ['native'],
    # 'add_reversed_seq': [True, False],
    # 'num_epochs': [30],  # no lower than 30
    # 'learning_rate': [0.005],  # no lower than 0.003
    # 'inner_size': [8],  # at least 6
    # 'num_layers': [2],  # at least 2
    # 'embed_size': [16],  # no lower than 16
    # 'initializer_range': [0.002],

    # ctn
    # 'add_with': [True],
    # 'strict_compositional': [False, True],
    # 'omit_type_2_verb_and_exp_theme': [False, True],


    # 'omit_type_2_verb_and_exp_theme': [False, True],
    # In the 'strict_compositional' condition, we removed 'preserve pepper', and kept everything else.
    # We think that the Transformer confused the novel input 'preserve pepper' with the familiar 'grow pepper'.
    # What if we keep 'preserve pepper' and remove 'grow pepper' instead?
    # If our hypothesis is true, the model should still select 'vinegar' for 'preserve pepper'.
    # Also, the model should confuse 'grow pepper' with 'preserve pepper',
    # and therefore it should incorrectly select 'vinegar' for 'grow pepper'.
    # Also, given what we are speculating now, when deleting 'preserve pepper' and testing on it,
    # we should test what the Transformer predicts for 'grow pepper'.
    # Due to our speculation, the model should still be able to get that right.
    # That is, only the performance of the removed item should be affected.



}

# ################################################## End of user-editable settings

if DSM_NAME == 'count':
    param2default_dsm = {
        # window size means how many neighbors are considered in forward direction
        'count_type': ('ww', 'summed', 4, 'linear'),  # currently, sentence-boundary is respected automatically
        'norm_type': None,  # None is slightly better than all others
        'reduce_type': ('svd', 30), # 30 performs best on exp 2b, but results in bad performance on other experiments
    }

elif DSM_NAME == 'random':
    param2default_dsm = {
        'embed_size': 32,
        'distribution': 'normal',
    }

elif DSM_NAME == 'w2v':
    param2default_dsm = {
        'w2vec_type': 'sg',                 # 'sg' performs better than 'cbow'
        'embed_size': 32,                   # 32 is best
        'window_size': 4,
        'num_epochs': 2,                    # 2 is best
        'initial_learning_rate': 0.02,      # 0.02 is best
    }

elif DSM_NAME == 'rnn':
    param2default_dsm = {
        # architecture
        'rnn_type': 'lstm',
        'embed_size': 64,           # 64 is better than any lower
        'num_layers': 2,
        # optimization
        'train_percent': 1.0,
        'embed_init_range': 0.05,   # 0.05 is good (but extremely robust against large changes)
        'dropout_prob': 0.0,        # must be 0.0 with num_layers=1
        'batch_size': 64,           # 64 is good
        'num_epochs': 4,            # more than 4 improves 1a and 2a accuracy, but 4 is best for 2b and 2c accuracy
        'learning_rate': 0.06,      # 0.06 with batch_size=64
        'grad_clip': 1.0,
        'lr_decay': 0.001,          # 0.001 but no larger
        'weight_decay': 0.0,        # keep at 0
        # evaluation
        'embeddings_location': 'wx',
    }

elif DSM_NAME == 'transformer':
    param2default_dsm = {
        # architecture
        'transformer_type': 'gpt2',
        'embed_size': 16,               # no lower than 16
        'inner_size': 4,                # at least 6
        'resid_pdrop': 0.0,             # 0 is best with lr=0.09
        'num_layers': 2,                # use 2 layers to test hypothesis that transformer learns tree structure
        'num_heads': 1,                 # 1 is best
        'seq_len': 8,                   # this must be larger than the largest sentence in the corpus
        # optimization
        'batch_size': 128,              # should be smaller than 576 (size of complete block)
        'num_epochs': 30,               # lower than 30 works well for num_layers=1, but not for num_layers=2
        'learning_rate': 0.005,         # is decayed during training with Adam optimizer
        'weight_decay': 0.0,            # 0.0 is best
        'adam_beta2': 0.999,            # default, robust to small changes
        'adam_epsilon': 1e-08,          # default, robust to small changes
        'label_smoothing': 0.0,         # default, robust to small changes
        'initializer_range': 0.002,     # 0.002 is best and is default

    }

elif DSM_NAME == 'lon':
    param2default_dsm = {

        # note: the LON is currently built from corpus - consecutive words are connected
        # todo: the co-occurrence matrix will be it's adjacency matrix,
        #  and the spreading activation functions on the adjacency matrix

        # 'count_type': ('ww', 'summed', 4, 'flat'),  # currently, sentence-boundary is respected automatically
        # 'norm_type': None,
        'excluded_tokens': None,
        'context_size': 1,
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
    'composition_fn': 'native',
}

param2default_corpus = {
    'complete_block': True,  # generate all possible samples, in addition to num_blocks, e.g. 576 for exp2
    'num_blocks': 400,  # 400 produces better loss in transformer than fewer blocks
    'include_location': False,
    'include_location_specific_agents': False,
    'add_with': True,
    'add_in': False,
    'strict_compositional': False,  # exclude type-3 verb + exp theme combinations from training
    #'omit_type_2_verb_and_exp_theme': False,  # exclude type-2 verb + exp theme combinations from training

    'add_reversed_seq': False,
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

# ################################################## Checks

if 'dropout_prob' in param2requests:
    for dp in param2requests['dropout_prob']:
        dp: float
        if dp > 0.0:
            if 'num_layers' in param2requests:
                if param2requests['num_layers'] == 1:
                    raise ValueError('Cannot use non-zero dropout when num_layers=1')
            else:
                if param2default['num_layers'] == 1:
                    raise ValueError('Cannot use non-zero dropout when num_layers=1')


if DSM_NAME == 'ctn':
    if 'composition_fn' in param2requests:
        for comp_fn in param2requests['composition_fn']:
            comp_fn: str
            if comp_fn != 'native':
                raise ValueError('CTN requires composition_fn=native')

if DSM_NAME == 'lon':
    if 'context_size' in param2requests:
        for context_size in param2requests['context_size']:
            context_size: int
            if context_size not in {1, 2}:
                raise ValueError('LON requires a context_size of 1 or 2')

if DSM_NAME == 'w2v':
    if 'composition_fn' in param2requests:
        for comp_fn in param2requests['composition_fn']:
            comp_fn: str
            if comp_fn == 'native':
                raise ValueError('Word2vec does not implement composition_fn=native')
    elif 'composition_fn' in param2default:
        if param2default['composition_fn'] == 'native':
            raise ValueError('Word2vec does not implement composition_fn=native')

if DSM_NAME == 'count':
    if 'composition_fn' in param2requests:
        for comp_fn in param2requests['composition_fn']:
            comp_fn: str
            if comp_fn == 'native':
                raise ValueError('Count models are not compatible with composition_fn=native')
    elif param2default['composition_fn'] == 'native':
        raise ValueError('Count models are not compatible with composition_fn=native')

# ################################################## End of checks


@dataclass
class CorpusParams:
    include_location: bool
    include_location_specific_agents: bool
    num_blocks: int
    complete_block: bool
    add_with: bool
    add_in: bool
    strict_compositional: bool

    add_reversed_seq: bool

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
    # evaluation
    embeddings_location: str

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
    initial_learning_rate: int

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})


@dataclass
class RandomControlParams:
    embed_size: int
    distribution: str

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
    context_size: Optional[int]  # only 1 or 2

    # count_type: Tuple[str, Optional[str], Optional[int], Optional[str]]
    # norm_type: Optional[str]  # e.g. None, 'row_sum', 'row_logentropy', 'tf_idf', 'ppmi'

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})


@dataclass
class Params:

    corpus_params: CorpusParams
    dsm: str
    dsm_params: Union[CountParams, RandomControlParams, Word2VecParams, RNNParams, TransformerParams]
    composition_fn: str
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
            dsm_params = RandomControlParams.from_param2val(tmp)
        elif param2val['dsm'] == 'w2v':
            dsm_params = Word2VecParams.from_param2val(tmp)
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
