from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math

from missingadjunct.corpus import Corpus
from missingadjunct.utils import make_blank_sr_df

from traindsms.utils import compose
from traindsms.params import Params
from traindsms.dsms.count import CountDSM
from traindsms.dsms.random_control import RandomControlDSM
from traindsms.dsms.w2vec import W2Vec
from traindsms.dsms.glove import GloVe
from traindsms.dsms.rnn import RNN
from traindsms.dsms.ctn import CTN
from traindsms.dsms.lon import LON


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

    # collect corpus data
    seq_num: List[List[int]] = []  # sequences of Ids
    seq_tok: List[List[str]] = []  # sequences of tokens
    seq_parsed: List[Tuple] = []  # sequences that are constituent-parsed
    for s in corpus.get_sentences():  # a sentence is a string
        tokens = s.split()
        seq_num.append([corpus.token2id[token] for token in tokens])
        seq_tok.append([token for token in tokens])
    for tree in corpus.get_trees():
        seq_parsed.append(tree)

    print(f'Number of sequences in corpus={len(seq_tok):,}')

    if params.dsm == 'count':
        dsm = CountDSM(params.dsm_params, corpus.vocab, seq_num)
    elif params.dsm == 'random':
        dsm = RandomControlDSM(params.dsm_params, corpus.vocab)
    elif params.dsm == 'w2v':
        dsm = W2Vec(params.dsm_params, corpus.vocab, seq_tok)
    elif params.dsm == 'glove':
        dsm = GloVe(params.dsm_params, corpus.vocab, seq_tok)
    elif params.dsm == 'rnn':
        dsm = RNN(params.dsm_params, corpus.vocab, seq_num)
    elif params.dsm == 'ctn':
        dsm = CTN(params.dsm_params, corpus.token2id, seq_parsed)
    elif params.dsm == 'lon':

        # TODO the LON does not need to inherit from network - no network code needed - only spreading_activation_analysis

        dsm = LON(params.dsm_params, seq_tok)  # TODO this requires adjacency matrix only (which is the co-mat)
    else:
        raise NotImplementedError

    # train
    dsm.train()
    print(f'Completed training the DSM', flush=True)

    # load evaluation df
    df_blank = make_blank_sr_df()
    df_results = df_blank.copy()
    instruments = df_blank.columns[3:]
    assert set(instruments).issubset(corpus.vocab)

    # fill in blank data frame with semantic-relatedness scores
    for verb_phrase, row in df_blank.iterrows():
        verb, theme = verb_phrase.split()
        scores = []
        for instrument in instruments:  # instrument columns start after the 3rd column

            # score spatial model
            if (not isinstance(dsm, CTN)) and not (isinstance(dsm, LON)):
                vp_e = compose(params.composition_fn, dsm.t2e[verb], dsm.t2e[theme])
                sr = cosine_similarity(vp_e[np.newaxis, :], dsm.t2e[instrument][np.newaxis, :]).item()

            # score CTN
            elif isinstance(dsm, CTN):
                if (verb, theme) in dsm.node_list:
                    sr_verb = dsm.activation_spreading_analysis(verb, dsm.node_list,
                                                                excluded_edges=[((verb, theme), theme)])
                    sr_theme = dsm.activation_spreading_analysis(theme, dsm.node_list,
                                                                 excluded_edges=[((verb, theme), verb)])
                else:
                    sr_verb = dsm.activation_spreading_analysis(verb, dsm.node_list, excluded_edges=[])
                    sr_theme = dsm.activation_spreading_analysis(theme, dsm.node_list, excluded_edges=[])
                sr = math.log(sr_verb[instrument] * sr_theme[instrument])


            else:
                raise NotImplementedError  # TOdo

            scores.append(sr)

            print(f'Relatedness between {verb_phrase:>22} and {instrument:>12} is {sr:.4f}', flush=True)

        # collect sr scores in new df
        df_results.loc[verb_phrase] = [row['verb-type'], row['theme-type'], row['phrase-type']] + scores

    print(df_results.loc['preserve pepper'].loc['vinegar'])
    print(df_results.loc['preserve pepper'].loc['dehydrator'])

    df_results.to_csv(save_path / 'df_sr.csv')

    return []

