from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

from src import config


def make_probe_simmat(w2e, probes, method):
    # filter probes
    num_probes = probes
    reduced_embeddings_mat = np.zeros((num_probes, config.Global.embed_size))
    for i in num_probes:
        reduced_embeddings_mat[i] = w2e[probes[i]]
    # sim
    if method == 'cosine':
        res = cosine_similarity(reduced_embeddings_mat)
    else:
        raise NotImplemented
    return res