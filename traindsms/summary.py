from pathlib import Path
import numpy as np
from typing import Tuple, Optional, Union, List
import pandas as pd
from scipy.stats import sem, t

from traindsms import config


def save_summary_to_txt(summary: Tuple[np.ndarray, np.ndarray, np.ndarray, str, Union[int, None]],
                        performance_name: str,
                        ) -> None:
    """
    output summary to text file.
    useful when plotting data with pgfplots on overleaf.org.

    notes:
        1. latex does not like spaces in file name
    """

    x, y_mean, h, label, job_id = summary

    # make path
    fn = performance_name + '_'
    fn += label.replace("\n", "-").replace(' ', '')
    path = config.Dirs.summaries / f'{fn}.txt'  # txt format makes file content visible on overleaf.org
    if not path.parent.exists():
        path.parent.mkdir()

    # save to text
    df = pd.DataFrame(data={'mean': y_mean, 'margin_of_error': h}, index=list(x))
    df.index.name = 'step'
    df.round(3).to_csv(path, sep=' ')

    print(f'Saved summary to {path}')
