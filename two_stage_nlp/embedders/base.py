from sklearn import preprocessing
import numpy as np
from cached_property import cached_property
import pickle

from sortedcontainers import SortedDict

from two_stage_nlp import config


class EmbedderBase(object):
    def __init__(self, param_name, job_name):
        self.param_name = param_name
        self.job_name = job_name
        self.w2e = dict()  # is populated by child class

    @property
    def location(self):
        res = config.Dirs.local_runs / self.param_name / self.job_name
        if not res.exists():
            res.mkdir(parents=True)
        return res

    # ///////////////////////////////////////////////////////////// w2e

    def save_w2e(self):
        p = self.location / 'embeddings.txt'
        with p.open('w') as f:
            for probe, embedding in sorted(self.w2e.items()):
                embedding_str = ' '.join(np.around(embedding, config.Embeddings.precision).astype(np.str).tolist())
                f.write('{} {}\n'.format(probe, embedding_str))

    def load_w2e(self):  # TODO unused - keep in case it is needed
        mat = np.loadtxt(config.Dirs.remote_runs / self.param_name / self.job_name / 'embeddings.txt',
                         dtype='str', comments=None)
        vocab = mat[:, 0]
        embed_mat = self.standardize_embed_mat(mat[:, 1:].astype('float'))
        self.w2e = self.embeds_to_w2e(embed_mat, vocab)

    # ///////////////////////////////////////////////////////////// corpus data

    @cached_property
    def vocab(self):
        p = config.Dirs.remote_root / '{}_{}_vocab.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
        if not p.exists():
            raise RuntimeError('{} does not exist'.format(p))
        #
        res = np.loadtxt(p, 'str').tolist()
        return res

    @cached_property
    def w2freq(self):
        p = config.Dirs.remote_root / '{}_w2freq.txt'.format(config.Corpus.name)
        if not p.exists():
            raise RuntimeError('{} does not exist'.format(p))
        #
        mat = np.loadtxt(p, dtype='str', comments=None)
        words = mat[:, 0]
        freqs = mat[:, 1].astype('int')
        res = {w: np.asscalar(f) for w, f in zip(words, freqs)}
        return res

    @cached_property
    def docs(self):
        p = config.Dirs.remote_root / '{}_{}_docs.pkl'.format(config.Corpus.name, config.Corpus.num_vocab)
        if not p.exists():
            raise RuntimeError('{} does not exist'.format(p))
        #
        with p.open('rb') as f:
            res = pickle.load(f)
        return res

    @cached_property
    def numeric_docs(self):
        p = config.Dirs.remote_root / '{}_{}_numeric_docs.pkl'.format(config.Corpus.name, config.Corpus.num_vocab)
        if not p.exists():
            raise RuntimeError('{} does not exist'.format(p))
        #
        with p.open('rb') as f:
            res = pickle.load(f)
        return res

    # ///////////////////////////////////////////////////////////// embeddings

    @staticmethod
    def standardize_embed_mat(mat):
        assert mat.shape[1] > 1
        scaler = preprocessing.StandardScaler()
        res = scaler.fit_transform(mat)
        return res

    @staticmethod
    def w2e_to_embeds(w2e):
        embeds = []
        for w in w2e.keys():
            embeds.append(w2e[w])
        res = np.vstack(embeds)
        if config.Eval.verbose:
            print('Converted w2e to matrix with shape {}'.format(res.shape))
        return res

    @staticmethod
    def embeds_to_w2e(embed_mat, vocab):
        res = SortedDict()
        for n, w in enumerate(vocab):
            res[w] = embed_mat[n]
        assert len(vocab) == len(res) == len(embed_mat)
        return res

    @property
    def dim1(self):
        res = next(iter(self.w2e.items()))[1].shape[0]
        return res


