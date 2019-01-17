import sys
import pandas as pd
import yaml

from src import config
from src.aggregator import Aggregator
from src.architectures import comparator
from src.evaluators.matching import Matching
from src.embedders.base import w2e_to_sims
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder


def aggregation_job(ev_name):
    print('Aggregating runs data for eval={}..'.format(ev_name))
    ag_matching = Aggregator(ev_name)
    df = ag_matching.make_df()
    p = config.Dirs.remote_root / '{}.csv'.format(ag_matching.ev_name)
    df.to_csv(p)
    print('Done. Saved aggregated data to {}'.format(ev_name, p))
    return df


def embedder_job(params2val_p):  # TODO put backup function from rnnlab to ludwigcluster, import it, and put it at end of job
    """
    Train a single embedder once, and evaluate all novice and expert scores for each task once
    """
    # params
    time_of_init = params2val_p.name  # TODO test
    with params2val_p.open('r') as f:
        param2val = yaml.load(f)
    print('===================================================')
    for k, v in param2val.items():
        print(k, v)
    # load embedder
    if 'random_type' in param2val:
        embedder = RandomControlEmbedder(param2val, time_of_init)
    elif 'rnn_type' in param2val:
        embedder = RNNEmbedder(param2val, time_of_init)
    elif 'w2vec_type' in param2val:
        embedder = W2VecEmbedder(param2val, time_of_init)
    elif 'count_type' in param2val:
        embedder = CountEmbedder(param2val,  time_of_init)
    elif 'glove_type' in param2val:
        raise NotImplementedError
    else:
        raise RuntimeError('Could not infer embedder name from param2val')
    # stage 1
    if not embedder.has_embeddings():  # in case previous job was interrupted after embedding completed
        print('Training...')
        embedder.train()
        embedder.save_w2freq()
        embedder.save_w2e()
    else:
        print('Found embeddings at {}'.format(embedder.location))
        embedder.load_w2e()  # in case previous job was interrupted during eval, and embeddings exist
    sys.stdout.flush()
    # stage 2
    for architecture in [comparator]:
        for ev in [
            Matching(architecture, 'cohyponyms', 'semantic'),
            Matching(architecture, 'cohyponyms', 'syntactic'),
            Matching(architecture, 'features', 'is'),
            Matching(architecture, 'features', 'has'),
            Matching(architecture, 'nyms', 'syn'),
            Matching(architecture, 'nyms', 'syn', suffix='_jw'),
            Matching(architecture, 'nyms', 'ant'),
            Matching(architecture, 'nyms', 'ant', suffix='_jw'),
            Matching(architecture, 'hypernyms'),
            Matching(architecture, 'events'),

            # Identification(architecture, 'nyms', 'syn', suffix=''),
            # Identification(architecture, 'nyms', 'ant', suffix=''),
        ]:
            if ev.suffix != '':
                print('WARNING: Using task file suffix "{}".'.format(ev.suffix))
            if not config.Eval.resample:
                print('WARNING: Not re-sampling data across replications.')
            # check scores_p
            scores_p = ev.make_scores_p(embedder.location)
            try:
                scores_p.parent.exists()
            except OSError:
                raise OSError('{} is not reachable. Check VPN or mount drive.'.format(scores_p))
            if scores_p.exists() and not config.Eval.debug:
                raise RuntimeError(
                    '{} should not exist. This is likely a failure of ludwigcluster to distribute tasks.')
            # make eval data - row_words can contain duplicates
            vocab_sims_mat = w2e_to_sims(embedder.w2e, embedder.vocab, embedder.vocab)
            all_eval_probes, all_eval_candidates_mat = ev.make_all_eval_data(vocab_sims_mat, embedder.vocab)
            ev.row_words, ev.col_words, ev.eval_candidates_mat = ev.downsample(
                all_eval_probes, all_eval_candidates_mat)
            if config.Eval.verbose:
                print('Shape of all eval data={}'.format(all_eval_candidates_mat.shape))
                print('Shape of down-sampled eval data={}'.format(ev.eval_candidates_mat.shape))
            #
            ev.pos_prob = ev.calc_pos_prob()
            # check that required embeddings exist for eval
            for scores_p in set(ev.row_words + ev.col_words):
                if scores_p not in embedder.w2e:
                    raise KeyError('"{}" required for evaluation "{}" is not in w2e.'.format(scores_p, ev.name))
            # score
            sims_mat = w2e_to_sims(embedder.w2e, ev.row_words, ev.col_words)  # sims can have duplicate rows
            ev.score_novice(sims_mat)
            df_rows = ev.train_and_score_expert(embedder)
            # save
            for df_row in df_rows:
                if config.Eval.verbose:
                    print('Saving score to {}'.format(scores_p.relative_to(config.Dirs.remote_root)))
                df = pd.DataFrame(data=[df_row], columns=['exp_score', 'nov_score'] + ev.df_header)
                if not scores_p.parent.exists():
                    scores_p.parent.mkdir(parents=True)
                with scores_p.open('a') as f:
                    df.to_csv(f, mode='a', header=f.tell() == 0, index=False)
            # figs
            if config.Eval.save_figs:
                ev.save_figs(embedder)
            print('-')