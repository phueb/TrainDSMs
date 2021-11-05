from pathlib import Path
import pandas as pd

from traindsms import config
from traindsms.params import Params


def main(param2val):
    """
    Train a single DSM once, and save results
    """

    # params
    params = Params.from_param2val(param2val)
    print(params)

    project_path = Path(param2val['project_path'])
    save_path = Path(param2val['save_path'])

    # in case job is run locally, we must create save_path
    if not save_path.exists():
        save_path.mkdir(parents=True)

    for k, v in param2val.items():
        print(k, v)

    dsm = init_dsm(param2val)  # todo decide which model to init

    # initialize dictionary for collecting performance data
    performance = {}
    eval_steps = 0  # TODO

    # todo train

    # collect performance in list of pandas series
    res = []
    for k, v in performance.items():
        if not v:
            continue
        df = pd.Series(v, index=eval_steps)
        df.name = k
        res.append(df)

    return res
