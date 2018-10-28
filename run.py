import pandas as pd
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
    (RandomControlEmbedder(param2ids, param2val) for param2ids, param2val in make_param2ids(RandomControlParams))
)

tasks = [
    NymMatching('noun', 'synonym'),  # TODO collapse noun and verb synonyms? (data is small)
    NymMatching('verb', 'synonym'),
    # Categorization('semantic'),
    # Categorization('syntactic')
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
    if config.Task.clear_scores:
        embedder.clear_scores()
    data = []
    index = []
    for task in tasks:
        if config.Task.retrain or not embedder.has_task(task.name):  # TODO task must exist config.num_reps times
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
            index.append('nov_' + task.name)
            data.append(task.score_novice(sims))
            index.append('exp_' + task.name)
            data.append(task.train_and_score_expert(embedder))
            # save scores
            scores = pd.Series(data=data, index=index)
            embedder.append_scores(scores)
            # figs
            task.save_figs(embedder.time_of_init)
        else:
            print('---------------------------------------------')
            print('Embedder has task "{}"'.format(task.name))
            print('---------------------------------------------')


# combine scores
scores_list = []
for p in config.Dirs.runs.rglob('scores.csv'):
    scores = pd.read_csv(p, header=None, squeeze=True, index_col=0)  # squeezes into series
    scores = scores.groupby(scores.index).mean()
    scores.name = p.parent.name
    scores_list.append(scores)
df = pd.concat(scores_list, axis=1)
df.index.name = 'task'
df.to_csv('all_scores.csv')
print(df)