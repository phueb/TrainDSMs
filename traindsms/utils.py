import pickle
import shutil

import numpy as np
import yaml
from sklearn.metrics.pairwise import cosine_similarity

from traindsms.embedders.rnn import RNNEmbedder
from traindsms.embedders.count import CountEmbedder
from traindsms.embedders.random_control import RandomControlEmbedder
from traindsms.embedders.w2vec import W2VecEmbedder
from traindsms import config


def move_scores_to_server(param2val, location):
    dst = config.RemoteDirs.runs / param2val['param_name']
    if not dst.exists():
        dst.mkdir(parents=True)
    shutil.move(str(location), str(dst))


def save_param2val(param2val, local=False):
    runs_dir = config.Dirs.runs if local else config.RemoteDirs.runs
    param2val_p = runs_dir / param2val['param_name'] / 'param2val.yaml'
    print('Saving param2val to {}'.format(param2val_p))
    if not param2val_p.exists():
        param2val['job_name'] = None
        with param2val_p.open('w', encoding='utf8') as f:
            yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)


def save_corpus_data(deterministic_w2f, vocab, docs, numeric_docs, skip_docs, num_vocab, local):
    #
    root = config.Dirs.root if local else config.RemoteDirs.root
    print('Sending results from corpus preprocessing to {}'.format(root))
    # save w2freq
    p = root / '{}_w2freq.txt'.format(config.Corpus.name)
    with p.open('w') as f:
        for probe, freq in deterministic_w2f.items():
            f.write('{} {}\n'.format(probe, freq))
    # save vocab
    p = root / '{}_{}_vocab.txt'.format(config.Corpus.name, num_vocab)
    with p.open('w') as f:
        for v in vocab:
            f.write('{}\n'.format(v))
    # save numeric_docs
    p = root / '{}_{}_numeric_docs.pkl'.format(config.Corpus.name, num_vocab)
    with p.open('wb') as f:
        pickle.dump(numeric_docs, f)
    # save docs
    if skip_docs or local:
        return  # takes long to upload docs to file server
    # save docs for each worker - otherwise multiple workers will open same files causing EOFError
    from ludwigcluster.config import SFTP
    for worker in SFTP.worker_names:
        p = root / '{}_{}_{}_docs.pkl'.format(worker, config.Corpus.name, num_vocab)
        with p.open('wb') as f:
            pickle.dump(docs, f)

    p = root / '{}_{}_docs.pkl'.format(config.Corpus.name, num_vocab)
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