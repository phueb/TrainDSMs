from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from missingadjunct.corpus import Corpus
from missingadjunct.utils import make_blank_sr_df

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

    save_path = Path(param2val['save_path'])

    # in case job is run locally, we must create save_path
    if not save_path.exists():
        save_path.mkdir(parents=True)

    corpus = Corpus(include_location=params.corpus_params.include_location,
                    include_location_specific_agents=params.corpus_params.include_location_specific_agents,
                    num_epochs=params.corpus_params.num_epochs,
                    seed=params.corpus_params.seed,
                    )

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
    embeddings = dsm.train()
    t2e = {t: e for t, e in zip(corpus.vocab, embeddings)}

    # load evaluation df
    df_blank = make_blank_sr_df()
    df_results = df_blank.copy()
    instruments = df_blank.columns[3:]
    assert set(instruments).issubset(corpus.vocab)

    if params.composition_fn == 'multiplication':
        composition_fn = lambda a, b: a * b
    else:
        raise NotImplementedError

    # fill in blank data frame with semantic-relatedness scores
    for verb_phrase, row in df_blank.iterrows():
        verb, theme = verb_phrase.split()
        scores = []
        for instrument in instruments:  # instrument columns start after the 3rd column
            vp_e = composition_fn(t2e[verb], t2e[theme])
            sr = cosine_similarity(vp_e[np.newaxis, :], t2e[instrument][np.newaxis, :]).item()
            scores.append(sr)

        # collect sr scores in new df
        df_results.loc[verb_phrase] = [row['verb-type'], row['theme-type'], row['phrase-type']] + scores

    print(df_results.round(3))

    df_results.to_csv(save_path / 'df_blank.csv')

    return []

