import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
from cached_property import cached_property
from collections import Counter

from src import config

nlp = spacy.load('en_core_web_sm')


class EmbedderBase(object):
    def __init__(self, corpus_name, name):
        self.corpus_name = corpus_name
        self.name = name

    @property
    def embeddings_fname(self):
        return '{}_{}.txt'.format(self.corpus_name, self.name)

    @property
    def corpus_fname(self):
        return '{}.txt'.format(self.corpus_name)

    @cached_property
    def corpus_data(self):
        return self.preprocess()  # token_ids, vocab

    @property
    def numeric_docs(self):
        return self.corpus_data[0]

    @property
    def vocab(self):
        return self.corpus_data[1]

    def preprocess(self):

        docs = []
        tokens = []
        # read corpus file
        p = config.Global.corpora_dir / self.corpus_fname
        texts = p.open('r').readlines()
        print('Tokenizing {} docs...'.format(len(texts)))
        # tokenize
        tokenizer = Tokenizer(nlp.vocab)
        for s_doc in tokenizer.pipe(texts, batch_size=50):  # creates spacy Docs
            doc = []
            for t in s_doc:
                token = t.text
                if token != '\n':
                    tokens.append(token)
                    doc.append(token)
            docs.append(doc)

        # if no vocag specified, use the whole corpus
        if config.Corpora.num_vocab is None:
            config.Corpora.num_vocab = len(set(tokens)) + 1

        # vocab
        print('Creating vocab...')
        w2f = Counter(tokens)
        vocab = sorted([config.Corpora.UNK] + [w for w, f in w2f.most_common(config.Corpora.num_vocab - 1)])
        print('Least frequent word occurs {} times'.format(
            np.min([f for w, f in w2f.most_common(config.Corpora.num_vocab - 1)])))
        print('Mapping words to ids...')
        # insert UNK + numericize
        t2id = {t: i for i, t in enumerate(vocab)}
        numeric_docs = []
        for doc in docs:
            numeric_doc = []

            for n, token in enumerate(doc):
                if token in t2id:
                    numeric_doc.append(t2id[token])
                else:
                    numeric_doc.append(t2id[config.Corpora.UNK])
            numeric_docs.append(numeric_doc)
        return numeric_docs, vocab

    def has_embeddings(self):
        p = config.Global.embeddings_dir / self.embeddings_fname
        return True if p.exists() else False

    @staticmethod
    def check_consistency(mat):  # TODO different models may produce embeddings that benefit from different expert hyperparameters
        # size check
        assert mat.shape[1] > 1
        # norm check
        assert np.max(mat) <= 1.0
        assert np.min(mat) >= -1.0
        print('Inf Norm of embeddings = {:.1f}'.format(np.linalg.norm(mat, np.inf)))

    def load_w2e(self):
        mat = np.loadtxt(config.Global.embeddings_dir / self.embeddings_fname, dtype='str', comments=None)
        embed_mat = mat[:, 1:].astype('float')
        embed_size = embed_mat.shape[1]
        w2e = {probe: embedding for probe, embedding in zip(mat[:, 0], embed_mat)}
        self.check_consistency(embed_mat)
        return w2e, embed_size

    def save(self, w2e):  # TODO serializing is faster (pickle, numpy)
        p = config.Global.embeddings_dir / self.embeddings_fname
        with p.open('w') as f:
            for probe, embedding in sorted(w2e.items()):
                embedding_str = ' '.join(embedding.astype(np.str).tolist())
                f.write('{} {}\n'.format(probe, embedding_str))

    def w2e_to_matrix(self, w2e):
        for key in w2e:
            vector = w2e[key]
            break
        num_rows = len(w2e)
        num_cols = len(vector)
        matrix = np.zeros([num_rows, num_cols], float)
        assert num_rows == len(self.vocab)
        for i in range(num_rows):
            matrix[i,:] = w2e[self.vocab[i]]
        return matrix

    def norm_rowsum(self, w2e):
        print('Normalizing matrix by row sums...')
        input_matrix = self.w2e_to_matrix(w2e)
        num_rows = len(input_matrix[:,0])
        num_cols = len(input_matrix[0,:])
        output_matrix = np.zeros([num_rows, num_cols], float)

        for i in range(num_rows):
            if input_matrix[i,:].sum() == 0:
                print('    Warning: Row {} ({}) had sum of zero. Setting prob to 0'.format(i, self.vocab[i]))
            else:
                output_matrix[i,:] = input_matrix[i,:] / input_matrix[i,:].sum()

        return output_matrix, num_cols

    def norm_colsum(self, w2e):
        print('Normalizing matrix by column sums...')
        input_matrix = self.w2e_to_matrix(w2e)
        num_rows = len(input_matrix[:,0])
        num_cols = len(input_matrix[0,:])
        output_matrix = np.zeros([num_rows, num_cols], float)

        for i in range(num_cols):
            if input_matrix[:,i].sum() == 0:
                print('    Warning: Column {} had sum of zero. Setting prob to 0'.format(i, self.vocab[i]))
            else:
                output_matrix[:,i] = input_matrix[:,i] / input_matrix[:,i].sum()

        return output_matrix, num_cols

    def norm_tdidf(self, w2e):
        print('Normalizing matrix by td-idf...')
        input_matrix = self.w2e_to_matrix(w2e)
        num_rows = len(input_matrix[:,0])
        num_cols = len(input_matrix[0,:])

        colprob_matrix = np.zeros([num_rows, num_cols], float)
        for i in range(num_cols):
            if input_matrix[:,i].sum() == 0:
                print('    Warning: Column {} had sum of zero. Setting prob to 0'.format(i, self.vocab[i]))
            else:
                colprob_matrix[:,i] = input_matrix[:,i] / input_matrix[:,i].sum()

        output_matrix = np.zeros([num_rows, num_cols], float)
        for i in range(num_rows):
            col_occ_count = np.count_nonzero(input_matrix[i,:]) + 1
            row_idf = float(num_cols) / col_occ_count
            for j in range(num_cols):
                output_matrix[i,j] = colprob_matrix[i,j] / row_idf

        return output_matrix, num_cols