"""
Performance of input embeddings of Transformer of type='gpt2' on task exp2b:
    acc=0.8 with lr=0.1 and epoch=100 and embed_size=8 and num_heads=4 and inner_size=16
    acc=0.6 with lr=0.1 and epoch=10  and embed_size=8 and num_heads=4 and inner_size=8
    acc=0.5 with lr=0.5 and epoch=10  and embed_size=8 and num_heads=4 and inner_size=16
    acc=0.5 with lr=0.2 and epoch=10  and embed_size=8 and num_heads=4 and inner_size=8





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
            ][2]

param2requests = {
    'seed': [0],
    # 'num_blocks': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'complete_block': [True],
    'learning_rate': [0.1],
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


elif DSM_NAME == 'transformer':
    param2default_dsm = {
        'transformer_type': 'gpt2',
        'embed_size': 8,
        'inner_size': 8,  # must be divisible by embed_size
        'dropout_prob': 0,
        'num_layers': 1,
        'num_heads': 4,
        'seq_len': 32,  # this must be larger than the largest sentence in the corpus
        'batch_size': 1,
        'num_epochs': 10,
        'learning_rate': 0.1,
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
    'composition_fn': 'multiplication',  # todo when "native" the function should be native to the model, e.g. next-word prediction with rnn
}

param2default_corpus = {
    'include_location': False,
    'include_location_specific_agents': False,
    'seed': 0,
    'num_blocks': 0,
    'complete_block': True,  # generate all possible samples, in addition to num_blocks, e.g. 576 for exp2
}

# avoid keys with the same name
for k in param2default_corpus:
    assert k not in param2default_dsm
param2default.update(param2default_corpus)
param2default.update(param2default_dsm)


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
class TransformerParams:
    transformer_type: str
    embed_size: int
    inner_size: int  # usually 4x larger than embed_size
    dropout_prob: float
    num_layers: int
    num_heads: int
    seq_len: int
    batch_size: int
    num_epochs: int
    learning_rate: float

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
            raise AttributeError('Invalid arg to "dsm".')

        corpus_params = CorpusParams.from_param2val(tmp)
        return cls(dsm=param2val['dsm'],
                   composition_fn=param2val['composition_fn'],
                   corpus_params=corpus_params,
                   dsm_params=dsm_params,
                   )
