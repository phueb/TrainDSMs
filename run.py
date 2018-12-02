from itertools import chain
import numpy as np

from src import config
from src.params import CountParams, RNNParams, Word2VecParams, RandomControlParams
from src.params import make_param2id
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder

from src.architectures.comparator import Comparator
from src.architectures.classifier import Classifier

from src.evals.identification import Identification
from src.evals.matching import Matching

from src.utils import w2e_to_sims


embedders = chain(
    (RNNEmbedder(param2id, param2val) for param2id, param2val in make_param2id(RNNParams)),
    (W2VecEmbedder(param2id, param2val) for param2id, param2val in make_param2id(Word2VecParams)),
    (RandomControlEmbedder(param2id, param2val) for param2id, param2val in make_param2id(RandomControlParams)),
    (CountEmbedder(param2id, param2val) for param2id, param2val in make_param2id(CountParams)),
)

# run full experiment
for architecture in [Comparator(), Classifier()]:
    # evals
    evals = [
        Matching(architecture, 'cohyponyms', 'semantic'),
        Matching(architecture, 'cohyponyms', 'syntactic'),
        Matching(architecture, 'nyms', 'syn'),  # TODO expert is well below novice
        Matching(architecture, 'nyms', 'ant'),
        Matching(architecture, 'hypernyms'),
        Matching(architecture, 'features', 'is'),
        Matching(architecture, 'features', 'has'),

        # Identification('nyms', 'syn'),  # TODO below chance - somethign is wrong
        # Identification('nyms', 'ant'),
        # Identification('cohyponyms', 'semantic'),  # TODO not enough lures
    ]
    # evaluate
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
                    for p in set(eval.all_row_words + eval.all_col_words):
                        if p not in embedder.w2e:
                            raise KeyError('"{}" required for eval "{}" is not in w2e.'.format(p, eval.name))
                    # shuffle + down sample (no need to shuffle between trials)
                    np.random.seed(rep_id)
                    eval.row_words = np.random.choice(eval.all_row_words,
                                                      size=min(len(eval.all_row_words), config.Eval.max_num_sim_dim),
                                                      replace=False).tolist()
                    eval.col_words = np.random.choice(eval.all_col_words,
                                                      size=min(len(eval.all_col_words), config.Eval.max_num_sim_dim),
                                                      replace=False).tolist()
                    # similarities
                    sims_mat = w2e_to_sims(embedder.w2e, eval.row_words, eval.col_words, config.Embeddings.sim_method)
                    print('Shape of similarity matrix: {}'.format(sims_mat.shape))
                    # score
                    eval.eval_candidates_mat = eval.make_eval_data(sims_mat)
                    eval.score_novice(sims_mat)
                    eval.train_and_score_expert(embedder, rep_id)
                    # figs
                    if config.Eval.save_figs:
                        eval.save_figs(embedder)
                else:
                    print('Embedder completed "{}" replication {}'.format(eval.name, rep_id))
                    print('---------------------------------------------')



