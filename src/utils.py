from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np


def make_probe_simmat(w2e, embed_size, probes, method):
    # filter probes
    num_probes = len(probes)
    reduced_embeddings_mat = np.zeros((num_probes, embed_size))
    for n, probe in enumerate(probes):
        reduced_embeddings_mat[n] = w2e[probe]
    # sim
    if method == 'cosine':
        res = cosine_similarity(reduced_embeddings_mat)
    else:
        raise NotImplemented  # TODO how to convert euclidian distance to sim measure?
    return res