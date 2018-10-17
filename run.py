import numpy as np
from itertools import chain

from src import config
from src.params import CountParams, RNNParams, Word2VecParams, RandomControlParams
from src.params import make_param2ids
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder

from src.tasks.categorization import Categorization
from src.tasks.nym_matching import NymMatching

from src.utils import w2e_to_sims

embedders = chain(
    (RNNEmbedder(param2ids, param2val) for param2ids, param2val in make_param2ids(RNNParams)),
    (CountEmbedder(param2ids, param2val) for param2ids, param2val in make_param2ids(CountParams)),
    (W2VecEmbedder(param2ids, param2val) for param2ids, param2val in make_param2ids(Word2VecParams)),
    (RandomControlEmbedder(param2ids, param2val) for param2ids, param2val in make_param2ids(RandomControlParams)))

tasks = [
    # NymMatching('noun', 'synonym'),
    # NymMatching('verb', 'synonym'),
    Categorization('semantic'),
    Categorization('syntactic')
]

# run full experiment
nov2scores = {}
exp2scores = {}
for embedder in embedders:
    # embed
    if config.Embeddings.retrain or not embedder.has_embeddings():
        print('Training embeddings')
        print('==========================================================================')
        embedder.train()
        if config.Embeddings.save:
            embedder.save_w2e()
            embedder.save_params()
            embedder.save_w2freq()
    else:
        print('Found embeddings at {}'.format(embedder.embeddings_fname))
        print('==========================================================================')
        embedder.load_w2e()

    # tasks
    for task in tasks:
        print('---------------------------------------------')
        print('Starting task "{}"'.format(task.name))
        print('---------------------------------------------')
        # check embeddings
        for p in set(task.row_words + task.col_words):
            if p not in embedder.w2e:
                # raise KeyError('"{}" required for task "{}" is not in w2e.'.format(p, task.name))
                print('"{}" required for task "{}" is not in w2e.'.format(p, task.name))
        # similarities
        sims = w2e_to_sims(embedder.w2e, task.row_words, task.col_words, config.Embeddings.sim_method)
        print('Shape of similarity matrix: {}'.format(sims.shape))
        # score
        nov2scores[embedder.time_of_init] = (task.name, task.score_novice(sims))
        exp2scores[embedder.time_of_init] = (task.name, task.train_and_score_expert(embedder))
        # figs
        task.save_figs(embedder.name)  # TODO save in runs dir with embeddings and params

# TODO save scores to csv
np.save('novice_scores.npy', nov2scores)
np.save('expert_scores.npy', exp2scores)