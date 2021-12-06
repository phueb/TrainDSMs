from typing import Optional, List, Tuple
from pathlib import Path

from ludwig.results import gen_param_paths


from traindsms import __name__, config
from traindsms.params import param2default, param2requests
from traindsms.dsms.rnn import RNN


LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH = None  # config.Dirs.runs if loading runs locally or None if loading data from ludwig

project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         isolated=True if RUNS_PATH is not None else False,
                                         runs_path=RUNS_PATH,
                                         #ludwig_data_path=LUDWIG_DATA_PATH,
                                         require_all_found=False,
                                         ):

    dsm = RNN.from_pretrained(param_path)
    print(dsm.model)

    embeddings = dsm.model.wx.weight.detach().numpy()
    print(embeddings.shape)
    print(embeddings)



    assert len(dsm.token2id) == len(embeddings)

    break