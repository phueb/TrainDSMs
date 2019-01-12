from itertools import chain
import sys

from src import config
from src.aggregator import Aggregator
from src.architectures import comparator
from src.evaluators.matching import Matching
from src.embedders.base import w2e_to_sims
from src.params import CountParams, RNNParams, Word2VecParams, RandomControlParams
from src.params import is_selected
from src.params import gen_combinations
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder


def aggregation_job(ev_name):
    ag_matching = Aggregator(ev_name)
    matching_df = ag_matching.make_df()
    matching_df.to_csv('{}.csv'.format(ag_matching.ev_name))
    print('Aggregated df for {} eval'.format(ev_name))


def embedder_job(embedder_name, params_df_row=None):
    embedders = chain(
        (W2VecEmbedder(param2id, param2val) for param2id, param2val in gen_combinations(Word2VecParams)
         if embedder_name in param2val.values() and is_selected(params_df_row, param2val)),

        (RNNEmbedder(param2id, param2val) for param2id, param2val in gen_combinations(RNNParams)
         if embedder_name in param2val.values() and is_selected(params_df_row, param2val)),

        (CountEmbedder(param2id, param2val) for param2id, param2val in gen_combinations(CountParams)
         if embedder_name in param2val.values() and is_selected(params_df_row, param2val)),

        (RandomControlEmbedder(param2id, param2val) for param2id, param2val in gen_combinations(RandomControlParams)
         if embedder_name in param2val.values() and is_selected(params_df_row, param2val))
    )
    runtime_errors = []
    while True:
        try:
            embedder = next(embedders)
        except RuntimeError as e:
            print('WARNING: embedder raised RuntimeError:')
            print(e)
            runtime_errors.append(e)
            continue
        except StopIteration:
            print('Finished embedding and evaluating {}'.format(embedder_name))
            for e in runtime_errors:
                print('with RunTimeError:')
                print(e)
            print()
            break
        #
        if config.Embeddings.retrain or not embedder.has_embeddings():
            print('Training...')
            print('==========================================================================')
            embedder.train()
            if config.Embeddings.save:
                embedder.save_params()
                embedder.save_w2freq()
                embedder.save_w2e()
        else:
            print('Found embeddings at {}'.format(embedder.location))
            print('==========================================================================')
            embedder.load_w2e()
        print('Embedding size={}'.format(embedder.w2e_to_embeds(embedder.w2e).shape[1]))
        sys.stdout.flush()
        # evaluate
        for architecture in [
            comparator,
            # classifier
        ]:
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
                for rep_id in range(config.Eval.num_reps):
                    if config.Eval.retrain or config.Eval.debug or not embedder.completed_eval(ev, rep_id):
                        print('Starting "{}" replication {}/{} with embedder={}'.format(
                            ev.full_name, rep_id + 1, config.Eval.num_reps, embedder.name))
                        if ev.suffix != '':
                            print('WARNING: Using task file suffix "{}".'.format(ev.suffix))
                        if not config.Eval.resample:
                            print('WARNING: Not re-sampling data across replications.')
                        print('---------------------------------------------')
                        # make eval data - row_words can contain duplicates
                        vocab_sims_mat = w2e_to_sims(embedder.w2e, embedder.vocab, embedder.vocab)
                        all_eval_probes, all_eval_candidates_mat = ev.make_all_eval_data(vocab_sims_mat, embedder.vocab)
                        ev.row_words, ev.col_words, ev.eval_candidates_mat = ev.downsample(
                            all_eval_probes, all_eval_candidates_mat,
                            seed=rep_id if config.Eval.resample else 42)
                        print('Shape of all eval data={}'.format(all_eval_candidates_mat.shape))
                        print('Shape of down-sampled eval data={}'.format(ev.eval_candidates_mat.shape))
                        #
                        ev.pos_prob = ev.calc_pos_prob()
                        # check embeddings for words
                        for p in set(ev.row_words + ev.col_words):
                            if p not in embedder.w2e:
                                raise KeyError('"{}" required for evaluation "{}" is not in w2e.'.format(p, ev.name))
                        # score
                        sims_mat = w2e_to_sims(embedder.w2e, ev.row_words, ev.col_words)  # sims can have duplicate rows
                        ev.score_novice(sims_mat)
                        ev.train_and_score_expert(embedder, rep_id)
                        # figs
                        if config.Eval.save_figs:
                            ev.save_figs(embedder)
                    else:
                        print('Embedder completed "{}" replication {}'.format(ev.full_name, rep_id))
                        print('---------------------------------------------')