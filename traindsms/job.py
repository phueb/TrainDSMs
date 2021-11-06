from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from missingadjunct.corpus import Corpus

from traindsms.params import Params
from traindsms.dsms.count import CountDSM
from traindsms.dsms.random_control import RandomControlDSM
from traindsms.dsms.w2vec import Word2Vec
from traindsms.dsms.glove import GloVe
from traindsms.dsms.rnn import RNN


def main(param2val):
    """
    Train a single DSM once, and save results
    """

    # params
    params = Params.from_param2val(param2val)
    print(params)

    project_path = Path(param2val['project_path'])
    save_path = Path(param2val['save_path'])

    # in case job is run locally, we must create save_path
    if not save_path.exists():
        save_path.mkdir(parents=True)

    corpus = Corpus(include_location=params.corpus_params.include_location,
                    include_location_specific_agents=params.corpus_params.include_location_specific_agents,
                    num_epochs=params.corpus_params.num_epochs,
                    seed=params.corpus_params.seed,
                    )
    corpus.print_counts()

    # convert tokens to IDs
    token2id = {t: n for n, t in enumerate(corpus.vocab)}
    sequences_numeric = []
    for s in corpus.get_sentences():  # a sentence is a string
        tokens = s.split()
        sequences_numeric.append([token2id[token] for token in tokens])

    if params.dsm == 'count':
        dsm = CountDSM(params.dsm_params, sequences_numeric, corpus.vocab)
    elif params.dsm == 'random':
        dsm = RandomControlDSM(params.dsm_params, sequences_numeric, corpus.vocab)
    elif params.dsm == 'w2v':
        dsm = Word2Vec(params.dsm_params, sequences_numeric, corpus.vocab)
    elif params.dsm == 'glove':
        dsm = GloVe(params.dsm_params, sequences_numeric, corpus.vocab)
    elif params.dsm == 'rnn':
        dsm = RNN(params.dsm_params, sequences_numeric, corpus.vocab)
    else:
        raise NotImplementedError

    # train
    co_mat = dsm.train()

    # show that similarity between "preserve pepper" is identical to "vinegar" and "dehydrator"
    v1 = co_mat[token2id['preserve']]
    v2 = co_mat[token2id['pepper']]
    vt = co_mat[token2id['vinegar']]
    vd = co_mat[token2id['dehydrator']]
    # v3 = v1 + v2
    v_norm = np.sqrt(sum(v2 ** 2))
    v3 = (np.dot(v1, v2) / v_norm ** 2) * v2  # project v1 on v2
    print(cosine_similarity(v3[np.newaxis, :], vt[np.newaxis, :]))
    print(cosine_similarity(v3[np.newaxis, :], vd[np.newaxis, :]))


    # todo fill in blank sr df and save to shared drive


    return []
