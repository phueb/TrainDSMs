from itertools import chain
import numpy as np

from src import config
from src.params import CountParams, RNNParams, Word2VecParams, RandomControlParams
from src.params import make_param2id
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder

from src.evals.classification import Classification  # TODO make classification an architecture not an evaluation
from src.evals.identification import Identification
from src.evals.matching import Matching

from src.utils import w2e_to_sims


embedders = chain(
    (RNNEmbedder(param2id, param2val) for param2id, param2val in make_param2id(RNNParams)),
    (W2VecEmbedder(param2id, param2val) for param2id, param2val in make_param2id(Word2VecParams)),
    (RandomControlEmbedder(param2id, param2val) for param2id, param2val in make_param2id(RandomControlParams)),
    (CountEmbedder(param2id, param2val) for param2id, param2val in make_param2id(CountParams)),
)


# a MATCHING eval consists of matching a probe with multiple correct answers
# an IDENTIFICATION eval consists of identifying correct answer from multiple-choice question

evals = [
    # Classification('cohyponyms', 'semantic'),  # TODO test multilabel graph
    # Classification('cohyponyms', 'syntactic'),
    # Identification('nyms', 'syn'),  # TODO below chance - somethign is wrong
    # Identification('nyms', 'ant'),
    # Identification('cohyponyms', 'semantic'),  # TODO not enough lures
    Matching('cohyponyms', 'semantic'),
    # Matching('cohyponyms', 'syntactic'),
    # Matching('nyms', 'syn'),
    # Matching('nyms', 'ant'),
    # Matching('hypernyms'),
    # Matching('features', 'is'),
    # Matching('features', 'has'),
]

# run full experiment
for embedder in embedders:
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
    # evals
    for eval in evals:
        for rep_id in range(config.Eval.num_reps):
            if config.Eval.retrain or not embedder.completed_task(eval, rep_id):
                print('Starting eval "{}" replication {}'.format(eval.name, rep_id))
                print('---------------------------------------------')
                # check runs
                for p in set(eval.row_words + eval.col_words):
                    if p not in embedder.w2e:
                        raise KeyError('"{}" required for eval "{}" is not in w2e.'.format(p, eval.name))
                # shuffle - only need to shuffle across reps (not across processes)
                np.random.seed(rep_id)
                np.random.shuffle(eval.row_words)
                np.random.seed(rep_id)
                np.random.shuffle(eval.col_words)    # TODO test

                # similarities
                eval_sims_mat = w2e_to_sims(embedder.w2e, eval.row_words, eval.col_words, config.Embeddings.sim_method)
                print('Shape of similarity matrix: {}'.format(eval_sims_mat.shape))
                # score
                eval.eval_candidates_mat = eval.make_eval_data(eval_sims_mat)
                eval.score_novice(eval_sims_mat)
                eval.train_and_score_expert(embedder, rep_id)
                # figs
                if config.Eval.save_figs:
                    eval.save_figs(embedder)
            else:
                print('Embedder completed "{}" replication {}'.format(eval.name, rep_id))
                print('---------------------------------------------')



