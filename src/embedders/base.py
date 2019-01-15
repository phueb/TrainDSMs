import sys
import pandas as pd
from collections import Counter, OrderedDict
from itertools import islice
from sklearn import preprocessing
from spacy.lang.en import English
import numpy as np
import pyprind
import yaml
import datetime
from cached_property import cached_property
from sklearn.metrics.pairwise import cosine_similarity
from sortedcontainers import SortedDict
from itertools import chain

from src import config


nlp = English()  # need this only for tokenization


class EmbedderBase(object):
    def __init__(self, param2val):
        self.param2val = param2val
        self.w2e = dict()  # is created by child class

    @cached_property
    def location(self):
        ps = chain(config.Dirs.runs.rglob('params.yaml'))
        while True:
            try:
                p = next(ps)
            except OSError:  # host is down
                raise OSError('Cannot access remote runs_dir. Check VPN and/or mount drive.')
            except StopIteration:
                break
            else:
                with p.open('r') as f:
                    param2val = yaml.load(f)
                    if param2val == self.param2val:
                        location = p.parent
                        return location
        # if location not found, create it
        time_of_init = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        res = config.Dirs.runs / time_of_init
        return res

    # ///////////////////////////////////////////////////////////// I/O

    @property
    def w2freq_fname(self):
        return '{}_w2freq.txt'.format(config.Corpus.name)

    def save_params(self):
        p = self.location / 'params.yaml'
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
        p = self.location / 'embeddings.txt'
        with p.open('w') as f:
            for probe, embedding in sorted(self.w2e.items()):
                embedding_str = ' '.join(np.around(embedding, config.Embeddings.precision).astype(np.str).tolist())
                f.write('{} {}\n'.format(probe, embedding_str))

    def load_w2e(self):
        mat = np.loadtxt(self.location / 'embeddings.txt', dtype='str', comments=None)
        vocab = mat[:, 0]
        embed_mat = self.standardize_embed_mat(mat[:, 1:].astype('float'))
        self.w2e = self.embeds_to_w2e(embed_mat, vocab)

    def completed_eval(self, ev, rep_id):
        num_total = len(ev.param2val_list)
        num_trained = 0
        p = ev.make_scores_p(self.location, rep_id)
        if p.exists():
            df = pd.read_csv(p, index_col=False)
            num_trained = len(df)
        print('{} rep {}: {}/{} param configurations completed'.format(
        ev.full_name, rep_id, num_trained, num_total))
        if num_trained == num_total:
            return True
        else:
            return False

    # ///////////////////////////////////////////////////////////// corpus data

    @classmethod
    def load_corpus_data(cls, num_vocab=config.Corpus.num_vocab):
        docs = []
        w2freq = Counter()
        # tokenize + count words
        p = config.Dirs.corpora / '{}.txt'.format(config.Corpus.name)
        with p.open('r') as f:
            texts = f.read().splitlines()  # removes '\n' newline character
        num_texts = len(texts)
        print('\nTokenizing {} docs...'.format(num_texts))
        pbar = pyprind.ProgBar(num_texts, stream=sys.stdout)

        # TODO tokenization could benefit from multiprocessing
        for text in texts:
            spacy_doc = nlp(text)
            doc = [w.text for w in spacy_doc]
            docs.append(doc)
            c = Counter(doc)
            w2freq.update(c)
            pbar.update()
        # vocab
        deterministic_w2f = OrderedDict(sorted(w2freq.items(), key=lambda item: (item[1], item[0]), reverse=True))
        if num_vocab is None:
            vocab = list(islice(deterministic_w2f.keys(), 0, num_vocab))
        else:
            vocab = list(islice(deterministic_w2f.keys(), 0, num_vocab - 1))
            vocab.append(config.Corpus.UNK)
        vocab = list(sorted(vocab))
        if num_vocab is None:  # if no vocab specified, use the whole corpus
            num_vocab = len(w2freq)
        print('Creating vocab of size {}...'.format(num_vocab))
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
        # save vocab - vocab needs to be overwritten when code in this function has been changed
        p = config.Dirs.corpora / '{}_{}_vocab.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
        if not p.exists():
            with p.open('w') as f:
                for v in vocab:
                    f.write('{}\n'.format(v))
        return numeric_docs, vocab, deterministic_w2f, docs

    @property
    def numeric_docs(self):
        return self.corpus_data[0]

    @property
    def vocab(self):
        p = config.Dirs.corpora / '{}_{}_vocab.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
        if p.exists():
            vocab = np.loadtxt(p, 'str').tolist()
            print('Loaded vocab from file. Found {} words.'.format(len(vocab)))
            # assert '.' in vocab
        else:
            print('Building vocab from corpus.')
            vocab = self.corpus_data[1]
        return vocab

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
        if (self.location / 'embeddings.txt').exists():  # TODO test
            return True
        else:
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


def w2e_to_sims(w2e, row_words, col_words):
    x = np.vstack([w2e[w] for w in row_words])
    y = np.vstack([w2e[w] for w in col_words])
    # sim
    res = cosine_similarity(x, y)
    print('Shape of similarity matrix: {}'.format(res.shape))
    return np.around(res, config.Embeddings.precision)