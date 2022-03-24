from typing import Optional, List, Tuple
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ludwig.results import gen_param_paths


from traindsms import __name__, config
from traindsms.params import param2default, param2requests
from traindsms.dsms.rnn import RNN


LUDWIG_DATA_PATH: Optional[Path] = None
RUNS_PATH = None  # config.Dirs.runs if loading runs locally or None if loading data from ludwig

project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         isolated=True if RUNS_PATH is not None else False,
                                         runs_path=RUNS_PATH,
                                         ludwig_data_path=LUDWIG_DATA_PATH,
                                         require_all_found=False,
                                         ):

    # load models and embeddings
    dsm = RNN.from_pretrained(param_path)
    embeddings = dsm.model.wx.weight.detach().numpy()
    assert len(dsm.token2id) == len(embeddings)

    # get similarities
    embedding_sims = cosine_similarity(embeddings)

    for token in ['cucumber', 'potato', 'pepper']:

        row_sims = embedding_sims[dsm.token2id[token]]
        for token_id in np.argsort(row_sims)[::-1][:6]:
            print(f'{token:<12} {dsm.id2token[token_id]:<12} sim={row_sims[token_id]}')
        print('-' * 30)



