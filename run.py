from itertools import chain

from src import config
from src.params import CountParams, RNNParams, Word2VecParams, RandomControlParams, GloveParams
from src.params import gen_all_param_combinations
from src.embedders.glove import GloveEmbedder
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder

from src.architectures import comparator
from src.architectures import classifier

from src.evaluators.identification import Identification
from src.evaluators.matching import Matching

from src.embedders.base import w2e_to_sims


# todo:
# TODO neighbors: make sure they don't reappear often to bias correct no-answers durign testing
# TODO neighbors: manually exclude neighbors which are actually synonyms or antonyms but not excluded automatically
# TODO adagrad: sgd underperforms adagrad - implement adagrad
# TODO unspecified cuda error

embedders = chain(
    (RNNEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(RNNParams)),
    (RandomControlEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(RandomControlParams)),
    (CountEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(CountParams)),
    (W2VecEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(Word2VecParams)),
    (GloveEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(GloveParams)),
)

# run full experiment
while True:
    # get embedder
    try:
        embedder = next(embedders)
    except RuntimeError as e:
        print('//////// WARNING: embedder raised RuntimeError:')
        print(e)
        continue
    except StopIteration:
        print('Finished experiment')
        break
    # embed
    if config.Embeddings.retrain or not embedder.has_embeddings():
        print('Training runs')
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
            Matching(architecture, 'nyms', 'ant'),
            Matching(architecture, 'hypernyms'),

            # Identification(architecture, 'nyms', 'syn'),  # TODO used to get above chance - what's wrong? -
            # TODO - maybe previously syns were only adjectives with adjectives, nouns with nouns, etc.
            # Identification(architecture, 'nyms', 'ant'),
            # Identification(architecture, 'cohyponyms', 'semantic'),  # TODO not enough lures - default to random lures below some threshold
        ]:
            for rep_id in range(config.Eval.num_reps):
                if config.Eval.retrain or not embedder.completed_eval(ev, rep_id):
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



