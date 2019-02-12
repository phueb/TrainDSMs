import pickle
import shutil

import numpy as np
import yaml
from sklearn.metrics.pairwise import cosine_similarity

from two_stage_nlp.embedders.rnn import RNNEmbedder
from two_stage_nlp.embedders.count import CountEmbedder
from two_stage_nlp.embedders.random_control import RandomControlEmbedder
from two_stage_nlp.embedders.w2vec import W2VecEmbedder
from two_stage_nlp import config


def move_scores_to_server(param2val, location):
    dst = config.Dirs.remote_runs / param2val['param_name']
    if not dst.exists():
        dst.mkdir(parents=True)
    shutil.move(str(location), str(dst))

    # write param2val to shared drive
    param2val_p = config.Dirs.remote_runs / param2val['param_name'] / 'param2val.yaml'
    if not param2val_p.exists():
        param2val['job_name'] = None
        with param2val_p.open('w', encoding='utf8') as f:
            yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)


def save_corpus_data(deterministic_w2f, vocab, docs, numeric_docs):
    # save w2freq
    p = config.Dirs.remote_root / '{}_w2freq.txt'.format(config.Corpus.name)
    with p.open('w') as f:
        for probe, freq in deterministic_w2f.items():
            f.write('{} {}\n'.format(probe, freq))
    # save vocab
    p = config.Dirs.remote_root / '{}_{}_vocab.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
    with p.open('w') as f:
        for v in vocab:
            f.write('{}\n'.format(v))
    # save numeric_docs
    p = config.Dirs.remote_root / '{}_{}_numeric_docs.pkl'.format(config.Corpus.name, config.Corpus.num_vocab)
    with p.open('wb') as f:
        pickle.dump(numeric_docs, f)
    # save docs
    p = config.Dirs.remote_root / '{}_{}_docs.pkl'.format(config.Corpus.name, config.Corpus.num_vocab)
    with p.open('wb') as f:
        pickle.dump(docs, f)


def init_embedder(param2val):
    if 'random_type' in param2val:
        return RandomControlEmbedder(param2val)
    elif 'rnn_type' in param2val:

        # TODO fix cuda error
        # raise NotImplementedError('Need to fix CUDA error before running RNNs')

        return RNNEmbedder(param2val)
    elif 'w2vec_type' in param2val:
        return W2VecEmbedder(param2val)
    elif 'count_type' in param2val:
        return CountEmbedder(param2val)
    elif 'glove_type' in param2val:
        raise NotImplementedError
    else:
        raise RuntimeError('Could not infer res name from param2val')


def w2e_to_sims(w2e, row_words, col_words):
    x = np.vstack([w2e[w] for w in row_words])
    y = np.vstack([w2e[w] for w in col_words])
    # sim
    res = cosine_similarity(x, y)
    if config.Eval.verbose:
        print('Shape of similarity matrix: {}'.format(res.shape))
    return np.around(res, config.Embeddings.precision)