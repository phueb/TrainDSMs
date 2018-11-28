import pandas as pd
import yaml
import numpy as np

from src import config

for embedder_class in ['rnn', 'count', 'w2vec']:
    for model_p in config.Dirs.runs.glob('*'):
        # check class
        with (model_p / 'params.yaml').open('r') as f:
            param2val = yaml.load(f)
        key = embedder_class + '_type'
        try:
            embedder_type = param2val[key]
            print('Found {}'.format(embedder_type))
        except KeyError:
            continue
        # loop over tasks
        largest_distrs_correct = []
        largest_distrs_shuffled = []
        task_names = []
        for task_p in model_p.glob('*/**'):  # only directories excluding model_p
            task_name = task_p.name
            task_names.append(task_name)
            print(task_name)
            # get score distributions (distributions are over replications)
            scores = pd.concat([pd.read_csv(p, index_col=False) for p in task_p.glob('scores_*.csv')], axis=1)
            bool_id_correct = (scores['shuffled'] == False).values[:, 0]
            bool_id_shuffled = (scores['shuffled'] == True).values[:, 0]
            distrs_chunk_correct = scores[bool_id_correct]['exp_score'].values
            distrs_chunk_shuffled = scores[bool_id_shuffled]['exp_score'].values
            # collect distribution with largest mean (over params)
            if not distrs_chunk_correct.size == 0:
                largest_distr_correct = distrs_chunk_correct[np.argmax(np.mean(distrs_chunk_correct, axis=1))]
                largest_distrs_correct.append(largest_distr_correct)
                print('correct')
                print(largest_distr_correct)
            if not distrs_chunk_shuffled.size == 0:
                largest_distr_shuffled = distrs_chunk_shuffled[np.argmax(np.mean(distrs_chunk_shuffled, axis=1))]
                largest_distrs_shuffled.append(largest_distr_shuffled)
                print('shuffled')
                print(largest_distr_shuffled)
        print()
    # make fig


