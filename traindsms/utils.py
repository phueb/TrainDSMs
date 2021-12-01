import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compose(fn: str,
            vector1: np.array,
            vector2: np.array,
            ) -> np.array:

    if fn == 'multiplication':
        return vector1 * vector2
    else:
        raise NotImplementedError


def calc_sr_cores_from_spatial_model(dsm, verb, theme, instruments, composition_fn):
    scores = []
    for instrument in instruments:
        vp_e = compose(composition_fn, dsm.t2e[verb], dsm.t2e[theme])
        sr = cosine_similarity(vp_e[np.newaxis, :], dsm.t2e[instrument][np.newaxis, :]).item()
        scores.append(sr)

    return scores
