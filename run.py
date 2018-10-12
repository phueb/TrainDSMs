import numpy as np

from src import config
from src.embedders.rnn import RNNEmbedder
from src.embedders.wd_matrix import WDEmbedder
from src.embedders.ww_matrix import WWEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.skipgram import SkipgramEmbedder

from src.tasks.categorization import Categorization
from src.tasks.nym_matching import NymMatching

from src.utils import w2e_to_sims
from src.utils import w2e_to_embeds

embedders = [
    RNNEmbedder(config.Corpus.name, 'srn'),
    RNNEmbedder(config.Corpus.name, 'lstm'),
    SkipgramEmbedder(config.Corpus.name),
    WDEmbedder(config.Corpus.name),
    WWEmbedder(config.Corpus.name),
    RandomControlEmbedder(config.Corpus.name)]
num_embedders = len(embedders)

tasks = [
    NymMatching('verb', 'synonym'),
    Categorization('semantic'),
    Categorization('syntactic')]
num_tasks = len(tasks)

# run full experiment
nov_scores_mat = np.zeros((num_embedders, num_tasks))
exp_scores_mat = np.zeros((num_embedders, num_tasks))
for i, embedder in enumerate(embedders):
    # embed
    if embedder.has_embeddings() and not config.Embeddings.retrain:
        print('==========================================================================')
        print('Found {}'.format(embedder.embeddings_fname))
        print('==========================================================================')
        w2e = embedder.load_w2e()
    else:
        print('==========================================================================')
        print('Did not find "{}" in {}.\nWill try to train "{}"'.format(
            embedder.embeddings_fname, config.Global.embeddings_dir, embedder.name))
        print('==========================================================================')
        w2e = embedder.train()
        if config.Embeddings.save:
            embedder.save(w2e)
    # tasks
    for j, task in enumerate(tasks):
        print('---------------------------------------------')
        print('Starting task "{}"'.format(task.name))
        print('---------------------------------------------')
        # check embeddings
        for p in set(task.sim_rows + task.sim_cols):
            if p not in w2e:
                # raise KeyError('Test-word "{}" in task "{}" is not in w2e.'.format(p, task.name))
                print('"{}" required for {} is not in {}.'.format(p, task.name, embedder.name))
        # similarities
        embeds = w2e_to_embeds(w2e)
        sims = w2e_to_sims(w2e, task.sim_rows, task.sim_cols, config.Global.sim_method)
        print('Shape of similarity matrix: {}'.format(sims.shape))
        # score
        nov_scores_mat[i, j] = task.score_novice(sims)
        exp_scores_mat[i, j] = task.train_and_score_expert(w2e, embeds.shape[1])
        # figs
        task.save_figs(embedder.name)
# save scores
np.save('novice_scores.npy', nov_scores_mat)
np.save('expert_scores.npy', exp_scores_mat)