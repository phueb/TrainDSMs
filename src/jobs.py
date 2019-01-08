from itertools import chain
import sys

from src import config
from src.architectures import comparator
from src.architectures import classifier
from src.evaluators.identification import Identification
from src.evaluators.matching import Matching
from src.embedders.base import w2e_to_sims
from src.params import CountParams, RNNParams, Word2VecParams, RandomControlParams, GloveParams
from src.params import gen_all_param_combinations
from src.embedders.glove import GloveEmbedder
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder


def embedder_job(embedder_name):
    embedders = chain(
        (W2VecEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(Word2VecParams)),
        (RNNEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(RNNParams)),
        (CountEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(CountParams)),
        (GloveEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(GloveParams)),
        (RandomControlEmbedder(param2id, param2val) for param2id, param2val in
         gen_all_param_combinations(RandomControlParams)),
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
            print('Finished experiment')
            for e in runtime_errors:
                print('with RunTimeError:')
                print(e)
            break
        else:
            if embedder.name != embedder_name:
                continue
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
            print('Found embeddings at {}'.format(config.Dirs.runs / embedder.time_of_init))
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
                        print('Starting evaluation "{}" replication {}'.format(ev.full_name, rep_id))
                        print('---------------------------------------------')
                        # make eval data - row_words can contain duplicates
                        vocab_sims_mat = w2e_to_sims(embedder.w2e, embedder.vocab, embedder.vocab)
                        all_eval_probes, all_eval_candidates_mat = ev.make_all_eval_data(vocab_sims_mat, embedder.vocab)
                        ev.row_words, ev.col_words, ev.eval_candidates_mat = ev.downsample(
                            all_eval_probes, all_eval_candidates_mat, rep_id)
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