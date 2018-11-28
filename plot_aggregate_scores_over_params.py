import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt

from src import config

INCLUDE_DICT = {'num_epochs': 20, 'num_vocab': 4096, 'corpus_name': 'childes-20180319'}
INCLUDE_DICT = {}  # TODO num epochs is not included in count params
TASK_CLASS = 'identification'

FONTSIZE = 24
WIDTH_PER_EMBEDDER = 4
HEIGHT = 6
DPI = 192

for embedder_class in [
    'rnn',
    'count',
    'w2vec',
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
        # exclude
        if not all([param2val[k] == v for k, v in INCLUDE_DICT.items()]):
            print('Excluding {}'.format(embedder_type))
            continue
        # collect data
        embedder_data = []
        for task_id, task_p in enumerate(embedder_p.glob('*/**')):  # only directories excluding model_p
            task_name = task_p.name
            if not TASK_CLASS in task_name:  # TODO make all tasks matching tasks
                continue
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
    title = '{} Performance on {} tasks'.format(embedder_class, TASK_CLASS)
    plt.title(title, fontsize=FONTSIZE)
    plt.xlabel(INCLUDE_DICT)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0, 1])
    # plot
    for embedder_id, embedder_data in enumerate(embedders_data):
        num_tasks = len(embedder_data)
        for task_id, (task_name, y_c, y_s) in enumerate(embedder_data):
            bw = 1 / (num_tasks + 1)
            x = embedder_id + (task_id / (num_tasks + 1))
            label = '_nolegend_' if embedder_id > 0 else task_name
            ax.bar(x, y_c.mean(), width=bw, yerr=y_c.std(), color=colors(task_id), label=label)
            ax.bar(x, y_s.mean(), width=bw, yerr=y_s.std(), color='black', alpha=0.5)
    # # label embedders
    # ax.set_xticks(np.arange(num_embedders))
    # ax.set_xticklabels(['\n'.join(['{}: {}'.format(k, v) for k, v in param2val.items()
    #                                if k not in INCLUDE_DICT.keys()])
    #                     for param2val in param2val_list], fontsize=FONTSIZE - 0.25 * FONTSIZE)
    # # legend
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=FONTSIZE)
    # plt.tight_layout()
    plt.show()
    raise SystemExit

    # TODO add novice score to plot


    # TODO each plot should vary along a singel parameter dimension only (instead of presenting all subtypes of a class)
    # TODO then one can stack figures in a powerpoint
    # TODO this also simolifies labeling of figures

