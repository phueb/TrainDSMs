from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sortedcontainers import SortedDict
from collections import Counter
from spacy.tokenizer import Tokenizer
import pyprind
import sys
import spacy

from src import config


nlp = spacy.load('en_core_web_sm')


def make_w2freq(corpus_name):
    w2f = Counter()
    # tokenize + count words
    p = config.Global.corpora_dir / '{}.txt'.format(corpus_name)
    with p.open('r') as f:
        texts = f.read().splitlines()  # removes '\n' newline character
    num_texts = len(texts)
    print('\nTokenizing {} docs...'.format(num_texts))
    tokenizer = Tokenizer(nlp.vocab)
    pbar = pyprind.ProgBar(num_texts, stream=sys.stdout)
    for spacy_doc in tokenizer.pipe(texts, batch_size=config.Corpus.spacy_batch_size):  # creates spacy Docs
        doc = [w.text for w in spacy_doc]
        c = Counter(doc)
        w2f.update(c)
        pbar.update()
    return w2f


def w2e_to_sims(w2e, sim_rows, sim_cols, method):  # TODO test
    x = np.vstack([w2e[w] for w in sim_rows])
    y = np.vstack([w2e[w] for w in sim_cols])
    # sim
    if method == 'cosine':
        res = cosine_similarity(x, y)
    else:
        raise NotImplemented  # TODO how to convert euclidian distance to sim measure?
    return res


def w2e_to_embeds(w2e):
    embeds = []
    for w in w2e.keys():
        embeds.append(w2e[w])
    res = np.vstack(embeds)
    print('Converted w2e to matrix with shape {}'.format(res.shape))
    return res


def matrix_to_w2e(input_matrix, vocab):
    res = SortedDict()
    for n, w in enumerate(vocab):
        res[w] = input_matrix[n]
    return res


def print_matrix(vocab, matrix, precision, row_list=None, column_list=None):

    t2id = {t: i for i, t in enumerate(vocab)}

    print()

    if row_list != None:
        for i in range(len(row_list)):
            if row_list[i] in t2id:
                row_index = t2id[row_list[i]]
                print('{:<15}   '.format(row_list[i]), end='')

                if column_list != None:
                    for j in range(len(column_list)):
                        if column_list[j] in t2id:
                            column_index = t2id[column_list[j]]
                        print('{val:6.{precision}f}'.format(precision=precision, val=matrix[row_index, column_index]), end='')
                    print()
                else:
                    for i in range(len(matrix[:, 0])):
                        print('{:<15}   '.format(vocab[i]), end='')
                        for j in range(len(matrix[i, :])):
                            print('{val:6.{precision}f}'.format(precision=precision, val=matrix[i, j]), end='')
                        print()
    else:
        for i in range(len(matrix[:, 0])):
            print('{:<15}   '.format(vocab[i]), end='')
            for j in range(len(matrix[i, :])):
                print('{val:6.{precision}f}'.format(precision=precision, val=matrix[i, j]), end='')
            print()