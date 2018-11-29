from itertools import chain

from src import config
from src.params import CountParams, RNNParams, Word2VecParams, RandomControlParams
from src.params import make_param2id
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder

from src.tasks.hypernym_identification import HypernymIdentification
from src.tasks.nym_identification import NymIdentification  # TODO make identification class
from src.tasks.matching import Matching

from src.utils import w2e_to_sims


embedders = chain(
    (RNNEmbedder(param2id, param2val) for param2id, param2val in make_param2id(RNNParams)),
    (CountEmbedder(param2id, param2val) for param2id, param2val in make_param2id(CountParams)),
    (W2VecEmbedder(param2id, param2val) for param2id, param2val in make_param2id(Word2VecParams)),
    (RandomControlEmbedder(param2id, param2val) for param2id, param2val in make_param2id(RandomControlParams))
)


# a MATCHING task consists of matching a probe with multiple correct answers
# an IDENTIFICATION task consists of identifying correct answer from multiple-choice question

tasks = [
    Matching('nyms', 'syn'),
    # Matching('nyms', 'ant'),
    # Matching('hypernyms'),
    # Matching('cohyponyms', 'semantic'),
    # Matching('cohyponyms', 'syntactic'),
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
    # tasks
    for task in tasks:
        for rep_id in range(config.Task.num_reps):
            if config.Task.retrain or not embedder.completed_task(task, rep_id):
                print('Starting task "{}" replication {}'.format(task.name, rep_id))
                print('---------------------------------------------')
                # check runs
                for p in set(task.row_words + task.col_words):
                    if p not in embedder.w2e:
                        raise KeyError('"{}" required for task "{}" is not in w2e.'.format(p, task.name))
                # similarities
                sims = w2e_to_sims(embedder.w2e, task.row_words, task.col_words, config.Embeddings.sim_method)
                print('Shape of similarity matrix: {}'.format(sims.shape))
                # score
                task.score_novice(sims)
                task.train_and_score_expert(embedder, rep_id)
                # figs
                if config.Task.save_figs:
                    task.save_figs(embedder)
            else:
                print('Embedder completed "{}" replication {}'.format(task.name, rep_id))
                print('---------------------------------------------')



