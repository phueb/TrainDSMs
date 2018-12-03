import sys
import pandas as pd
from collections import Counter, OrderedDict
from itertools import islice
from sklearn import preprocessing
import spacy
import numpy as np
import pyprind
import yaml
import datetime
from cached_property import cached_property
from sklearn.metrics.pairwise import cosine_similarity
from sortedcontainers import SortedDict

from spacy.tokenizer import Tokenizer

from src import config


nlp = spacy.load('en_core_web_sm')


class EmbedderBase(object):
    def __init__(self, param2val):
        self.param2val = param2val
        self.w2e = dict()  # is created by child class
        self.time_of_init = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') \
            if not self.has_embeddings() else self.has_embeddings()

    # ///////////////////////////////////////////////////////////// I/O

    @property
    def w2freq_fname(self):
        return '{}_w2freq.txt'.format(config.Corpus.name)

    def save_params(self):
        p = config.Dirs.runs / self.time_of_init / 'params.yaml'
        if not p.parent.exists():
            p.parent.mkdir()
        with p.open('w', encoding='utf8') as outfile:
            yaml.dump(self.param2val, outfile, default_flow_style=False, allow_unicode=True)

    def save_w2freq(self):
        p = config.Dirs.corpora / self.w2freq_fname
        with p.open('w') as f:
            for probe, freq in self.w2freq.items():
                f.write('{} {}\n'.format(probe, freq))

    def save_w2e(self):
        p = config.Dirs.runs / self.time_of_init / 'embeddings.txt'
        with p.open('w') as f:
            for probe, embedding in sorted(self.w2e.items()):
                embedding_str = ' '.join(np.around(embedding, config.Embeddings.precision).astype(np.str).tolist())
                f.write('{} {}\n'.format(probe, embedding_str))

    def load_w2e(self):
        mat = np.loadtxt(config.Dirs.runs / self.time_of_init / 'embeddings.txt', dtype='str', comments=None)
        vocab = mat[:, 0]
        embed_mat = self.standardize_embed_mat(mat[:, 1:].astype('float'))
        self.w2e = self.embeds_to_w2e(embed_mat, vocab)

    def completed_task(self, task, rep_id):
        p = config.Dirs.runs / self.time_of_init / task.name / 'scores_{}.csv'.format(rep_id)
        num_total = len(task.param2val_list)
        num_trained = 0
        if p.exists():
            df = pd.read_csv(p, index_col=False)
            num_trained = len(df)
        print('---------------------------------------------')
        print('Replication {}: Training for {}/{} param configurations completed'.format(
            rep_id, num_trained, num_total))
        if num_trained == num_total:
            return True
        else:
            return False

    # ///////////////////////////////////////////////////////////// corpus data

    @classmethod
    def load_corpus_data(cls, num_vocab=config.Corpus.num_vocab):
        docs = []
        w2freq = Counter()
        # tokenize + count.py words
        p = config.Dirs.corpora / '{}.txt'.format(config.Corpus.name)
        with p.open('r') as f:
            texts = f.read().splitlines()  # removes '\n' newline character
        num_texts = len(texts)
        print('\nTokenizing {} docs...'.format(num_texts))
        tokenizer = Tokenizer(nlp.vocab)
        pbar = pyprind.ProgBar(num_texts, stream=sys.stdout)
        for spacy_doc in tokenizer.pipe(texts, batch_size=config.Corpus.spacy_batch_size):  # creates spacy Docs
            doc = [w.text for w in spacy_doc]
            docs.append(doc)
            c = Counter(doc)
            w2freq.update(c)
            pbar.update()
        # vocab
        if num_vocab is None:  # if no vocab specified, use the whole corpus
            num_vocab = len(w2freq) + 1
        print('Creating vocab of size {}...'.format(num_vocab))
        deterministic_w2f = OrderedDict(sorted(w2freq.items(), key=lambda item: (item[1], item[0]), reverse=True))
        vocab = list(islice(deterministic_w2f.keys(), 0, num_vocab - 1))
        vocab.append(config.Corpus.UNK)
        print('Least frequent word occurs {} times'.format(deterministic_w2f[vocab[-2]]))
        assert '\n' not in vocab
        assert len(vocab) == num_vocab
        # insert UNK + make numeric
        print('Mapping words to ids...')
        t2id = {t: i for i, t in enumerate(vocab)}
        numeric_docs = []
        for doc in docs:
            numeric_doc = []

            for n, token in enumerate(doc):
                if token in t2id:
                    numeric_doc.append(t2id[token])
                else:
                    doc[n] = config.Corpus.UNK
                    numeric_doc.append(t2id[config.Corpus.UNK])
            numeric_docs.append(numeric_doc)
        return numeric_docs, vocab, deterministic_w2f, docs

    @property
    def numeric_docs(self):
        return self.corpus_data[0]

    @property
    def vocab(self):
        return self.corpus_data[1]

    @property
    def w2freq(self):
        if self.has_embeddings():
            p = config.Dirs.corpora / self.w2freq_fname
            mat = np.loadtxt(p, dtype='str', comments=None)
            words = mat[:, 0]
            freqs = mat[:, 1].astype('int')
            res = {w: np.asscalar(f) for w, f in zip(words, freqs)}
        else:
            res = self.corpus_data[2]
        return res

    @property
    def docs(self):
        return self.corpus_data[3]  # w2vec needs this

    @cached_property
    def corpus_data(self):
        return self.load_corpus_data()

    # ///////////////////////////////////////////////////////////// embeddings

    def has_embeddings(self):
        for p in config.Dirs.runs.rglob('params.yaml'):
            with p.open('r') as f:
                param2val = yaml.load(f)
                if param2val == self.param2val:
                    embed_p = p.parent / 'embeddings.txt'
                    if embed_p.exists():
                        time_of_init = p.parent
                        return time_of_init
        return False

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
        print('Converted w2e to matrix with shape {}'.format(res.shape))
        return res

    @staticmethod
    def embeds_to_w2e(embeds, vocab):
        res = SortedDict()
        for n, w in enumerate(vocab):
            res[w] = embeds[n]
        assert len(vocab) == len(res) == len(embeds)
        return res

    @property
    def dim1(self):
        res = next(iter(self.w2e.items()))[1].shape[0]
        return res


def w2e_to_sims(w2e, row_words, col_words):
    x = np.vstack([w2e[w] for w in row_words])
    y = np.vstack([w2e[w] for w in col_words])
    # sim
    res = cosine_similarity(x, y)
    print('Shape of similarity matrix: {}'.format(res.shape))
    return np.around(res, config.Embeddings.precision)