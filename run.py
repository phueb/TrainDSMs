from itertools import chain

from src import config
from src.params import CountParams, RNNParams, Word2VecParams, RandomControlParams
from src.params import make_param2id
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder

from src.tasks.category_label_detection import CatLabelDetection
from src.tasks.category_member_verification import CatMEmberVer
from src.tasks.nym_detection import NymDetection

from src.utils import w2e_to_sims


embedders = chain(
    (W2VecEmbedder(param2id, param2val) for param2id, param2val in make_param2id(Word2VecParams)),
    (RNNEmbedder(param2id, param2val) for param2id, param2val in make_param2id(RNNParams)),
    (CountEmbedder(param2id, param2val) for param2id, param2val in make_param2id(CountParams)),
    (RandomControlEmbedder(param2id, param2val) for param2id, param2val in make_param2id(RandomControlParams))
)


# a verification task consists of 2 words that either belong together or not.
# a detection task consists of a word and multiple candidate words where only one candidate belongs with test word.

tasks = [
    CatLabelDetection('semantic'),
    # CatLabelDetection('syntactic'),  # TODO what should labels be ? action, thing, property?
    NymDetection('antonym'),
    NymDetection('synonym'),
    CatMEmberVer('semantic'),
    CatMEmberVer('syntactic'),
]

# TODO parallelize across tasks? celery: each expert training is a job

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
            if config.Task.retrain or not embedder.has_task(task, rep_id):
                print('---------------------------------------------')
                print('Starting task "{}"'.format(task.name))
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
                print('---------------------------------------------')
                print('Embedder has task "{}"'.format(task.name))
                print('---------------------------------------------')



