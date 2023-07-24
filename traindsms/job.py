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
from traindsms.dsms.rnn import RNN
from traindsms.dsms.transformer import Transformer
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
                    seed=param2val['job_name'],
                    add_with=params.corpus_params.add_with,
                    add_in=params.corpus_params.add_in,
                    strict_compositional=params.corpus_params.strict_compositional,
                    )

    # load blank df for evaluating sr scores
    df_blank = make_blank_sr_df()
    df_results = df_blank.copy()
    instruments = df_blank.columns[4:]  # instrument columns start after the 4th column
    if not set(instruments).issubset(corpus.vocab):
        raise RuntimeError('Not all instruments in corpus. Add more blocks or set complete_block=True')

    # collect corpus data
    seq_num: List[List[int]] = []  # sequences of Ids
    seq_tok: List[List[str]] = []  # sequences of tokens
    seq_parsed: List[Tuple] = []  # sequences that are constituent-parsed
    for s in corpus.get_sentences():  # a sentence is a string
        tokens = s.split()
        seq_num.append([corpus.token2id[token] for token in tokens])  # numeric (token IDs)
        seq_tok.append(tokens)  # raw tokens
        if params.corpus_params.add_reversed_seq:
            seq_num.append([corpus.token2id[token] for token in tokens][::-1])
            seq_tok.append(tokens[::-1])
    for tree in corpus.get_trees():
        seq_parsed.append(tree)

    # save corpus text to disk
    with open(save_path / 'corpus.txt', 'w') as f:
        for s in corpus.get_sentences():
            f.write(s + '\n')

    print('Corpus Seed: ', corpus.seed)
    print(f'Number of sequences in corpus={len(seq_tok):,}', flush=True)

    if params.dsm == 'count':
        dsm = CountDSM(params.dsm_params, corpus.vocab, seq_num)
    elif params.dsm == 'random':
        dsm = RandomControlDSM(params.dsm_params, corpus.vocab)
    elif params.dsm == 'w2v':
        dsm = W2Vec(params.dsm_params, corpus.vocab, seq_tok)
    elif params.dsm == 'rnn':
        dsm = RNN(params.dsm_params, corpus.token2id, seq_num, df_blank, instruments, save_path)
    elif params.dsm == 'transformer':
        dsm = Transformer(params.dsm_params, corpus.token2id, seq_num, df_blank, instruments, save_path, corpus.eos)
    elif params.dsm == 'ctn':
        dsm = CTN(params.dsm_params, corpus.token2id, seq_parsed)
    elif params.dsm == 'lon':
        dsm = LON(params.dsm_params, seq_tok)  # TODO the net is built directly from corpus rather than co-occ
    else:
        raise NotImplementedError

    # train
    dsm.train()
    print(f'Completed training the DSM', flush=True)

    # fill in blank data frame with semantic-relatedness scores
    for verb_phrase, row in df_blank.iterrows():
        verb, theme = verb_phrase.split()

        # score graphical models
        if isinstance(dsm, LON) or isinstance(dsm, CTN):
            scores = dsm.calc_sr_scores(verb, theme, instruments)

        # score spatial models
        else:
            if params.composition_fn == 'native':  # use next-word prediction to compute sr scores
                scores = dsm.calc_native_sr_scores(verb, theme, instruments)
            else:
                scores = calc_sr_cores_from_spatial_model(dsm, verb, theme, instruments, params.composition_fn)

        # collect sr scores in new df
        df_results.loc[verb_phrase] = [row['verb-type'],
                                       row['theme-type'],
                                       row['phrase-type'],
                                       row['location-type']
                                       ] + scores

    df_results.to_csv(save_path / 'df_sr.csv')

    # prepare collected data for returning to Ludwig
    performance = dsm.get_performance()
    series_list = []
    for k, v in performance.items():
        if k == 'epoch':
            continue
        s = pd.Series(v, index=performance['epoch'])
        s.name = k
        series_list.append(s)

    # save model
    if isinstance(dsm, Transformer):
        dsm.model.save_pretrained(str(save_path))

    print('Completed main.job.', flush=True)

    return series_list

