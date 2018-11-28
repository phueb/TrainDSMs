import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt

from src import config

WIDTH_PER_EMBEDDER = 4
HEIGHT = 6
DPI = 192

for embedder_class in [
    'w2vec',
    'rnn',
    'count',
]:
    embedders_data = []
    param2val_list = []
    for embedder_p in config.Dirs.runs.glob('*'):
        # check class
        with (embedder_p / 'params.yaml').open('r') as f:
            param2val = yaml.load(f)
        key = embedder_class + '_type'
        try:
            embedder_type = param2val[key]
            print('Found {}'.format(embedder_type))
        except KeyError:
            continue
        # collect data
        embedder_data = []
        for task_id, task_p in enumerate(embedder_p.glob('*/**')):  # only directories excluding model_p
            task_name = task_p.name
            # get score distributions (distributions are over replications)
            scores = pd.concat([pd.read_csv(p, index_col=False) for p in task_p.glob('scores_*.csv')], axis=1)
            bool_id_correct = (scores['shuffled'] == False).values[:, 0]
            bool_id_shuffled = (scores['shuffled'] == True).values[:, 0]
            distrs = scores[bool_id_correct]['exp_score'].values
            distrs_s = scores[bool_id_shuffled]['exp_score'].values
            if distrs.size == 0 or distrs_s.size == 0:
                continue
            # get distribution with largest mean (over params)
            largest_distr_c = distrs[np.argmax(np.mean(distrs, axis=1))]  # correct input-output mapping
            largest_distr_s = distrs_s[np.argmax(np.mean(distrs_s, axis=1))]  # shuffled input-output mapping
            assert len(largest_distr_c) == len(largest_distr_s)
            embedder_data.append((task_name, largest_distr_c, largest_distr_s))
        embedders_data.append(embedder_data)
        param2val_list.append(param2val)
    # fig
    colors = plt.cm.get_cmap('tab10')
    num_embedders = len(embedders_data)
    fig, ax = plt.subplots(figsize=(WIDTH_PER_EMBEDDER * num_embedders, HEIGHT), dpi=DPI)
    plt.title(embedder_class)
    # plot
    for embedder_id, embedder_data in enumerate(embedders_data):
        num_tasks = len(embedder_data)
        for task_id, (task_name, y_c, y_s) in enumerate(embedder_data):
            x = embedder_id + (task_id / num_tasks)
            print(x)
            label = '_nolegend_' if embedder_id > 0 else task_name
            bw = 1 / num_tasks
            ax.bar(x, y_c.mean(), width=bw, yerr=y_c.std(), color=colors(task_id), label=label)
            ax.bar(x, y_s.mean(), width=bw, yerr=y_s.std(), color='black', alpha=0.5)
    # label embedders
    ax.set_xticks(np.arange(num_embedders))
    ax.set_xticklabels(['\n'.join(['{}: {}'.format(k, v) for k, v in param2val.items()])
                        for param2val in param2val_list])

    plt.legend(loc='upper left')  # TODO put outside plot
    plt.show()
    raise SystemExit

