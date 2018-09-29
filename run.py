import numpy as np

from src import config
from src.tasks.cat_classification import CatClassification
from src.embedders.lstm import LSTMEmbedder
from src.embedders.wd_matrix import WDEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.skipgram import SkipgramEmbedder
from src.embedders.ww_matrix import WWEmbedder
from src.utils import make_probe_simmat

embedders = [
    # LSTMEmbedder(config.Corpora.name),
    # WWEmbedder(config.Corpora.name, 'terms_v4096_ws0_wt0_ww0_nnone_rnone0'),
    # WWEmbedder(config.Corpora.name, 'terms_v4096_ws0_wt0_ww0_nnone_rsvd200'),
    WDEmbedder(config.Corpora.name, 'terms_v4096_ws0_wt0_ww0_nlogentropy_rsvd512'),
    WWEmbedder(config.Corpora.name),
    RandomControlEmbedder(config.Corpora.name),
    # WWEmbedder(config.Corpora.name, 'terms_v4096_ws0_wt0_ww0_nlogentropy_rnone0'),
    # WWEmbedder(config.Corpora.name, 'terms_v4096_ws0_wt0_ww0_nlogentropy_rsvd200'),

    SkipgramEmbedder(config.Corpora.name),
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
        w2e, embed_size = embedder.load_w2e()
    else:
        print('==========================================================================')
        print('Did not find "{}" in {}.\nWill try to train "{}"'.format(
            embedder.embeddings_fname, config.Global.embeddings_dir, embedder.name))
        print('==========================================================================')
        w2e, embed_size = embedder.train()
        if config.Embeddings.save:
            embedder.save(w2e)
    # tasks
    for j, task in enumerate(tasks):  # TODO different probes for each task?
        print('---------------------------------------------')
        print('Starting task "{}"'.format(task.name))
        print('---------------------------------------------')
        # check embeddings
        excluded = []
        for p in task.probes:
            if p not in w2e:
                raise KeyError('Probe "{}" in task "{}" is not in w2e.'.format(p, task.name))
        # similarities
        probe_simmat = make_probe_simmat(w2e, embed_size, task.probes, config.Global.sim_method)
        print('Number of probes in similarity matrix: {}'.format(len(probe_simmat)))
        # score
        nov_scores_mat[i, j] = task.score_novice(probe_simmat)  # TODO force common API between tasks
        exp_scores_mat[i, j] = task.train_and_score_expert(w2e, embed_size)
        # figs
        task.save_figs(embedder.name)
# save scores
# noinspection PyTypeChecker
np.savetxt('novice_scores.txt', nov_scores_mat)  # TODO tune experts by maximizing average performance across ALL models
# noinspection PyTypeChecker
np.savetxt('expert_scores.txt', exp_scores_mat)