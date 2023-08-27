import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compose(fn: str,
            vector1: np.array,
            vector2: np.array,
            ) -> np.array:

    if fn == 'multiplication':
        return vector1 * vector2
    elif fn == 'addition':
        return vector1 * vector2
    else:
        raise NotImplementedError


def calc_sr_cores_from_spatial_model(dsm, verb, theme, instruments, composition_fn):
    """
    Calculate semantic relatedness scores for a verb and theme using a spatial model.

    Note:
        2 vectors are added or multiplied together to form a verb-phrase vector.
        Then, the verb-phrase vector is compared to each instrument vector to get a score.
    """
    scores = []
    for instrument in instruments:
        vp_e = compose(composition_fn, dsm.t2e[verb], dsm.t2e[theme])
        sr = cosine_similarity(vp_e[np.newaxis, :], dsm.t2e[instrument][np.newaxis, :]).item()
        scores.append(sr)

    return scores


def calc_sr_cores_from_spatial_model_componential(dsm, verb, theme, instruments):
    """
    Calculate semantic relatedness scores for a verb and theme using a spatial model.

    Note:
        Each vector is compared to each instrument vector to get a score.
        Then, the 2 scores are multiplied together to get a single score.
    """

    scores = []
    for instrument in instruments:
        sr_1 = cosine_similarity(dsm.t2e[verb][np.newaxis, :], dsm.t2e[instrument][np.newaxis, :]).item()
        sr_2 = cosine_similarity(dsm.t2e[theme][np.newaxis, :], dsm.t2e[instrument][np.newaxis, :]).item()
        sr = sr_1 * sr_2
        scores.append(sr)

    return scores
