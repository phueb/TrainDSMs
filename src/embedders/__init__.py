import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
from cached_property import cached_property
from collections import Counter
import pyprind
import sys
from sortedcontainers import SortedDict

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
        with p.open('r') as f:
            texts = f.readlines()
            print('\nTokenizing {} docs...'.format(len(texts)))
            # tokenize
            tokenizer = Tokenizer(nlp.vocab)
            pbar = pyprind.ProgBar(len(texts), stream=sys.stdout)
            for s_doc in tokenizer.pipe(texts, batch_size=50):  # creates spacy Docs
                doc = []
                for t in s_doc:
                    token = t.text
                    if token != '\n':
                        tokens.append(token)
                        doc.append(token)
                docs.append(doc)
                pbar.update()
        # if no vocab specified, use the whole corpus
        if config.Corpora.num_vocab is None:
            config.Corpora.num_vocab = len(set(tokens)) + 1

        # vocab
        print('Creating vocab of size {}...'.format(config.Corpora.num_vocab))
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
        w2e = SortedDict({probe: embedding for probe, embedding in zip(mat[:, 0], embed_mat)})
        self.check_consistency(embed_mat)
        return w2e

    def save(self, w2e):  # TODO serializing is faster (pickle, numpy)
        p = config.Global.embeddings_dir / self.embeddings_fname
        with p.open('w') as f:
            for probe, embedding in sorted(w2e.items()):
                embedding_str = ' '.join(embedding.astype(np.str).tolist())
                f.write('{} {}\n'.format(probe, embedding_str))

    def norm_rowsum(self, input_matrix):
        print('\nNormalizing matrix by row sums...')
        num_rows = len(input_matrix[:,0])
        num_cols = len(input_matrix[0,:])
        output_matrix = np.zeros([num_rows, num_cols], float)
        pbar = pyprind.ProgBar(num_rows, stream=sys.stdout)
        for i in range(num_rows):
            if input_matrix[i,:].sum() == 0:
                print('    Warning: Row {} ({}) had sum of zero. Setting prob to 0'.format(i, self.vocab[i]))
            else:
                output_matrix[i,:] = input_matrix[i,:] / input_matrix[i,:].sum()
            pbar.update()
        return output_matrix, num_cols

    def norm_colsum(self, input_matrix):
        print('\nNormalizing matrix by column sums...')
        num_rows = len(input_matrix[:,0])
        num_cols = len(input_matrix[0,:])
        output_matrix = np.zeros([num_rows, num_cols], float)
        pbar = pyprind.ProgBar(num_cols, stream=sys.stdout)
        for i in range(num_cols):
            if input_matrix[:,i].sum() == 0:
                print('    Warning: Column {} had sum of zero. Setting prob to 0'.format(i, self.vocab[i]))
            else:
                output_matrix[:,i] = input_matrix[:,i] / input_matrix[:,i].sum()
            pbar.update()
        return output_matrix, num_cols

    def norm_tdidf(self, input_matrix):
        print('\nNormalizing matrix by td-idf...')
        num_rows = len(input_matrix[:,0])
        num_cols = len(input_matrix[0,:])
        print('Calculating column probs')
        pbar = pyprind.ProgBar(num_cols, stream=sys.stdout)
        colprob_matrix = np.zeros([num_rows, num_cols], float)
        for i in range(num_cols):
            if input_matrix[:,i].sum() == 0:
                print('    Warning: Column {} had sum of zero. Setting prob to 0'.format(i, self.vocab[i]))
            else:
                colprob_matrix[:,i] = input_matrix[:,i] / input_matrix[:,i].sum()
            pbar.update()
        print('Calculating td-idf scores')
        output_matrix = np.zeros([num_rows, num_cols], float)
        pbar = pyprind.ProgBar(num_rows, stream=sys.stdout)
        for i in range(num_rows):
            col_occ_count = np.count_nonzero(input_matrix[i,:]) + 1
            row_idf = float(num_cols) / col_occ_count
            for j in range(num_cols):
                output_matrix[i,j] = colprob_matrix[i,j] / row_idf
            pbar.update()
        return output_matrix, num_cols

    def norm_ppmi(self, input_matrix):
        print('\nNormalizing matrix by ppmi')
        num_rows = len(input_matrix[:,0])
        num_cols = len(input_matrix[0,:])

        row_sums = input_matrix.sum(1)
        col_sums = input_matrix.sum(0)
        matrix_sum = row_sums.sum()

        output_matrix = np.zeros([num_rows, num_cols], float)
        pbar = pyprind.ProgBar(num_rows, stream=sys.stdout)
        for i in range(num_rows):
            for j in range(num_cols):
                if input_matrix[i, j] == 0:
                    output_matrix[i, j] = 0
                elif (row_sums[i] == 0) or (col_sums[j] == 0):
                    output_matrix[i, j] = 0
                else:
                    top = input_matrix[i, j] / matrix_sum
                    bottom = (row_sums[i] / matrix_sum) * (col_sums[j] / matrix_sum)
                    div = top/bottom
                    if div <= 1:
                        output_matrix[i, j] = 0
                    else:
                        output_matrix[i, j] = np.log(div)
            pbar.update()
        return output_matrix, num_cols

    def norm_logentropy(self, input_matrix):
        print('\nNormalizing matrix by log entropy')
        num_rows = len(input_matrix[:,0])
        num_cols = len(input_matrix[0,:])
        output_matrix = np.zeros([num_rows, num_cols], float)

        print('Computing row probabilities')
        row_prob_matrix = np.zeros([num_rows, num_cols], float)
        pbar = pyprind.ProgBar(num_rows, stream=sys.stdout)
        for i in range(num_rows):
            if input_matrix[i,:].sum() == 0:
                print('    Warning: Row {} ({}) had sum of zero. Setting prob to 0'.format(i, self.vocab[i]))
            else:
                row_prob_matrix[i,:] = input_matrix[i,:] / input_matrix[i,:].sum()
            pbar.update()

        print('Computing entropy scores')
        log_freqs = np.log(input_matrix + 1)
        pbar = pyprind.ProgBar(num_rows, stream=sys.stdout)
        for i in range(num_rows):
            row_entropy = np.dot(row_prob_matrix[i, :], np.log(row_prob_matrix[i, :] + 1))
            global_weight = 1 + (row_entropy / np.log(num_cols + 1))

            for j in range(num_cols):
                output_matrix[i, j] = log_freqs[i, j] * global_weight
            pbar.update()

        return output_matrix, num_cols

    def reduce_svd(self, input_matrix, dimensions=config.Reduce.dimensions):
        print('\nReducing matrix using SVD to {} singular values'.format(dimensions))
        u, s, v = np.linalg.svd(input_matrix)
        reduced_matrix = u[:, 0:dimensions]
        return reduced_matrix, dimensions

    def reduce_rva(self, input_matrix, length=config.Reduce.dimensions, mean=config.Reduce.rv_mean, stdev=config.Reduce.rv_stdev):
        print('\nReducing matrix using RVA')
        num_rows = len(input_matrix[:, 0])
        num_cols = len(input_matrix[0, :])
        random_vectors = np.random.normal(mean,stdev,[num_rows,length])
        rva_matrix = np.zeros([num_rows, length], float)
        pbar = pyprind.ProgBar(num_rows, stream=sys.stdout)
        for i in range(num_rows):
            for j in range(num_rows):
                rva_matrix[i,:] += (input_matrix[i,j]*random_vectors[j,:])
            pbar.update()

        return rva_matrix, length

