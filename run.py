import numpy as np

from src import config
from src.tasks.cat_classification import CatClassification
from src.embedders.lstm import LSTMEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.skipgram import SkipgramEmbedder
from src.utils import make_probe_simmat


embedders = [LSTMEmbedder(config.Corpora.name),
             SkipgramEmbedder(config.Corpora.name),
             RandomControlEmbedder(config.Corpora.name)]
tasks = [CatClassification('semantic'),
         CatClassification('syntactic')]

# initialize
num_embedders = len(embedders)
num_tasks = len(tasks)
nov_scores_mat = np.zeros((num_embedders, num_tasks))
exp_scores_mat = np.zeros((num_embedders, num_tasks))

# train and score novices and experts on task_data
for i, embedder in enumerate(embedders):
    # embed
    if embedder.has_embeddings() and not config.Embeddings.retrain:
        print('Found {} in {}'.format(embedder.embeddings_fname, embedder.embeddings_dir))
        w2e = embedder.load_w2e()
    else:
        print('Did not find {} in {}. Will try to train.'.format(embedder.embeddings_fname, embedder.embeddings_dir))
        w2e = embedder.train()
        if config.Embeddings.save:
            embedder.save(w2e)
    # tasks
    for j, task in enumerate(tasks):  # TODO different probes for each task?
        # check embeddings
        for p in task.probes:
            if p not in w2e.keys():
                raise KeyError('Probe "{}" in task "{}" is not in w2e.'.format(p, task.name))
        # similarities
        probe_simmat = make_probe_simmat(w2e, task.probes, config.Global.sim_method)
        # score
        nov_scores_mat[i, j] = task.score_novice(probe_simmat)  # TODO force common API between tasks
        exp_scores_mat[i, j] = task.train_and_score_expert(w2e)
        # figs
        task.save_figs(w2e)

# save scores
# noinspection PyTypeChecker
np.savetxt('novice_scores.txt', nov_scores_mat)
# noinspection PyTypeChecker
np.savetxt('expert_scores.txt', exp_scores_mat)