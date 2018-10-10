import numpy as np

from src import config
from src.tasks.cat_classification import CatClassification
from src.embedders.rnn import RNNEmbedder
from src.embedders.wd_matrix import WDEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.skipgram import SkipgramEmbedder
from src.embedders.ww_matrix import WWEmbedder
from src.utils import matrix_to_simmat
from src.utils import w2e_to_matrix

embedders = [
    RNNEmbedder(config.Corpora.name, 'srn'),
    RNNEmbedder(config.Corpora.name, 'lstm'),
    SkipgramEmbedder(config.Corpora.name),
    #WDEmbedder(config.Corpora.name),
    #WWEmbedder(config.Corpora.name),

    RandomControlEmbedder(config.Corpora.name)]
num_embedders = len(embedders)

tasks = [
    CatClassification('semantic'),
    CatClassification('syntactic')]
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
        excluded = []
        for p in task.probes:
            if p not in w2e:
                raise KeyError('Probe "{}" in task "{}" is not in w2e.'.format(p, task.name))
        # similarities
        mat = w2e_to_matrix(w2e, task.probes)
        probe_simmat = matrix_to_simmat(mat, config.Global.sim_method)
        print('Number of probes in similarity matrix: {}'.format(len(probe_simmat)))
        # score
        nov_scores_mat[i, j] = task.score_novice(probe_simmat)  # TODO force common API between tasks
        exp_scores_mat[i, j] = task.train_and_score_expert(w2e, mat.shape[1])
        # figs
        task.save_figs(embedder.name)
# save scores
np.save('novice_scores.npy', nov_scores_mat)
np.save('expert_scores.npy', exp_scores_mat)