import numpy as np
from cytoolz import itertoolz
from io import StringIO

from src.embedders import EmbedderBase


class MatrixEmbedder(EmbedderBase):
    def __init__(self, corpus_name, name_suffix):
        super().__init__(corpus_name, '{}'.format(name_suffix))


WINDOW_SIZE = 5
WINDOW_WEIGHT = 'flat'
UNK = "UNKNOWN"


LINES = StringIO('''The child likes1 toy1 . The child likes toy2 . The child likes toy3 .
The man eats food1 . The man eats food2 . The man eats food3 .
The woman reads book1 . The woman reads book2 . The woman reads book3 .''')


def prepare(d):
    res = []
    for n, token in enumerate(d):
        if token in t2id:
            res.append(t2id[token])
        else:
            res.append(t2id[UNK])
    return res


def update_matrix(mat, ids):
    windows = itertoolz.sliding_window(WINDOW_SIZE, ids)
    for w in windows:
        for t1_id, t2_id, dist in zip([w[0]] * (WINDOW_SIZE - 1), w[1:], range(1, WINDOW_SIZE)):
            if WINDOW_WEIGHT == "linear":
                mat[t1_id, t2_id] += WINDOW_SIZE - dist
            elif WINDOW_WEIGHT == "flat":
                mat[t1_id, t2_id] += 1

# pre-processing
vocab = set()
docs = []
for line in LINES:
    # build docs
    doc = (line.strip().strip('\n').strip()).split()
    docs.append(doc)
    # build vocab
    for t in doc:
        vocab.add(t)
vocab.add(UNK)
vocab = sorted(vocab)
t2id = {t: i for i, t in enumerate(vocab)}
vocab_size = len(vocab)

# build matrix
cooc_mat = np.zeros([vocab_size, vocab_size], int)
for doc in docs:
    token_ids = prepare(doc)  # insert unknowns and convert to id - separated from update_matrix for readability
    update_matrix(cooc_mat, token_ids)
print(cooc_mat)

# verify
num_windows = 0
for doc in docs:
    num_windows += (len(doc) - WINDOW_SIZE + 1)
num_coocs_per_window = WINDOW_SIZE - 1
print(np.sum(cooc_mat), num_windows * num_coocs_per_window)  # only works when using "flat"
