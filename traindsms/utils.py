import numpy as np


def compose(fn: str,
            vector1: np.array,
            vector2: np.array,
            ) -> np.array:

    if fn == 'multiplication':
        return vector1 * vector2
    else:
        raise NotImplementedError
