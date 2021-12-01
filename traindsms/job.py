from pathlib import Path
from typing import List, Tuple
import pandas as pd

from missingadjunct.corpus import Corpus
from missingadjunct.utils import make_blank_sr_df

from traindsms.utils import calc_sr_cores_from_spatial_model
from traindsms.params import Params
from traindsms.dsms.count import CountDSM
from traindsms.dsms.random_control import RandomControlDSM
from traindsms.dsms.w2vec import W2Vec
from traindsms.dsms.glove import GloVe
from traindsms.dsms.rnn import RNN
from traindsms.dsms.transformer import TransformerDSM
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
                    num_epochs=params.corpus_params.num_blocks,
                    complete_epoch=params.corpus_params.complete_block,
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

    print(f'Number of sequences in corpus={len(seq_tok):,}', flush=True)

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
    elif params.dsm == 'transformer':
        dsm = TransformerDSM(params.dsm_params, corpus.token2id, seq_num, output_dir=str(save_path))
    elif params.dsm == 'ctn':
        dsm = CTN(params.dsm_params, corpus.token2id, seq_parsed)
    elif params.dsm == 'lon':
        dsm = LON(params.dsm_params, seq_tok)  # TODO the net is built directly from corpus rather than co-occ
    else:
        raise NotImplementedError

    # train
    dsm.train()
    print(f'Completed training the DSM', flush=True)

    # load evaluation df
    df_blank = make_blank_sr_df()
    df_results = df_blank.copy()
    instruments = df_blank.columns[3:]  # instrument columns start after the 3rd column
    assert set(instruments).issubset(corpus.vocab)

    # fill in blank data frame with semantic-relatedness scores
    for verb_phrase, row in df_blank.iterrows():
        verb, theme = verb_phrase.split()

        # score graphical models
        if isinstance(dsm, LON) or isinstance(dsm, CTN):
            scores = dsm.calc_sr_scores(verb, theme, instruments)

        # score spatial models
        else:
            scores = calc_sr_cores_from_spatial_model(dsm, verb, theme, instruments, params.composition_fn)

        # collect sr scores in new df
        df_results.loc[verb_phrase] = [row['verb-type'], row['theme-type'], row['phrase-type']] + scores

    df_results.to_csv(save_path / 'df_sr.csv')

    # prepare collected data for returning to Ludwig
    performance = dsm.get_performance()
    s = pd.Series(performance['eval_loss'], index=performance['epoch'])
    s.name = 'eval_loss'

    print('Completed main.job.', flush=True)

    return [s]

