import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
import copy

from src import config

EMBEDDER_CLASS = 'count'
EMBEDDER_TYPE = None  # can be None

INCLUDE_DICT = {'num_vocab': 4096, 'corpus_name': 'childes-20180319'}
TASK_CLASS = 'matching'

LEG1_Y = 1.3
NUM_LEG_COLS = 6
Y_STEP_SIZE = 0.25
TITLE_FONTSIZE = 24
LEG_FONTSIZE = 12
AX_FONTSIZE = 12
WIDTH_PER_EMBEDDER = 5
HEIGHT = 10
DPI = 192 / 2


def add_double_legend(lines_list, labels1, labels2):
    leg1 = plt.legend([l[0] for l in lines_list], labels1, loc='upper center',
                      bbox_to_anchor=(0.5, LEG1_Y), ncol=NUM_LEG_COLS, frameon=False, fontsize=LEG_FONTSIZE)
    for lines in lines_list:
        for line in lines:
            line.set_color('black')
    plt.legend(lines_list[0], labels2, loc='upper center',
               bbox_to_anchor=(0.5, LEG1_Y - 0.2), ncol=3, frameon=False, fontsize=LEG_FONTSIZE)
    plt.gca().add_artist(leg1)  # order of legend creation matters here


# collect data for plotting
embedders_data = []
param2val_list = []
for embedder_p in config.Dirs.runs.glob('*'):
    # check embedder_class and type
    with (embedder_p / 'params.yaml').open('r') as f:
        param2val = yaml.load(f)
    key = EMBEDDER_CLASS + '_type'
    try:
        embedder_type = param2val[key]
    except KeyError:
        continue
    else:
        if EMBEDDER_TYPE is not None and embedder_type != EMBEDDER_TYPE:
            continue
        else:
            print('Found {}'.format(embedder_type))
    # exclude
    if not all([param2val[k] == v for k, v in INCLUDE_DICT.items()]):
        print('Excluding {}'.format(embedder_type))
        continue
    # collect data
    embedder_data = []
    for task_id, task_p in enumerate(embedder_p.glob('*/**')):  # only directories excluding model_p
        task_name = task_p.name
        if TASK_CLASS not in task_name:
            continue
        # get score distributions (distributions are over replications)
        scores = pd.concat([pd.read_csv(p, index_col=False) for p in task_p.glob('scores_*.csv')], axis=1)
        bool_id_correct = (scores['shuffled'] == False).values[:, 0]
        bool_id_shuffled = (scores['shuffled'] == True).values[:, 0]
        distrs = scores[bool_id_correct]['exp_score'].values
        distrs_s = scores[bool_id_shuffled]['exp_score'].values
        score_nov = scores['nov_score'].values[0, 0]
        if distrs.size == 0 or distrs_s.size == 0:
            continue
        # get distribution with largest mean (over params)
        largest_distr_c = distrs[np.argmax(np.mean(distrs, axis=1))]  # correct input-output mapping
        largest_distr_s = distrs_s[np.argmax(np.mean(distrs_s, axis=1))]  # shuffled input-output mapping
        assert len(largest_distr_c) == len(largest_distr_s)
        embedder_data.append((task_name, largest_distr_c, largest_distr_s, score_nov))
    if embedder_data:
        embedders_data.append(embedder_data)
        param2val_list.append(param2val)

# fig
colors = plt.cm.get_cmap('tab10')
num_embedders = len(embedders_data)
fig, ax = plt.subplots(figsize=(WIDTH_PER_EMBEDDER * num_embedders, HEIGHT), dpi=DPI)
# fig, ax = plt.subplots(figsize=(12, HEIGHT), dpi=DPI)
embedder = EMBEDDER_TYPE or EMBEDDER_CLASS
title = '{} Performance on {} tasks'.format(embedder, TASK_CLASS)
plt.title(title, fontsize=TITLE_FONTSIZE, y=1.3)
plt.xlabel(INCLUDE_DICT, fontsize=AX_FONTSIZE)
if TASK_CLASS == 'matching':
    ylabel = 'Balanced Accuracy'
    ylims = [0.5, 1]
    yticks = np.arange(0.5, 1 + Y_STEP_SIZE, Y_STEP_SIZE)
elif TASK_CLASS == 'identification':
    ylabel = 'Accuracy'
    ylims = [0, 1]
    yticks = np.arange(0, 1 + Y_STEP_SIZE, Y_STEP_SIZE)
else:
    raise AttributeError('Invalid arg to "TASK_CLASS".')
plt.ylabel(ylabel, fontsize=AX_FONTSIZE)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(ylims)
# plot
lines = []
labels1 = []
for embedder_id, embedder_data in enumerate(embedders_data):
    num_tasks = len(embedder_data)
    lines = []
    for task_id, (task_name, y_e, y_s, y_n) in enumerate(embedder_data):
        bw = (1 / (num_tasks + 1)) / 3  # 3 due to expert, shuffled, novice
        x = embedder_id + (task_id / (num_tasks + 1))
        if embedder_id > 0:
            labels1.append(task_name.replace('_', '\n'))
        l1, = ax.bar(x + 0 * bw, y_e.mean(), width=bw, yerr=y_e.std(), color=colors(task_id))
        l2, = ax.bar(x + 1*bw, y_s.mean(), width=bw, yerr=y_s.std(), color=colors(task_id), alpha=0.75)
        l3, = ax.bar(x + 2*bw, y_n, width=bw, color=colors(task_id), alpha=0.5)
        lines.append((copy.copy(l1), copy.copy(l2), copy.copy(l3)))
# xtick labels
ax.set_xticks(np.arange(num_embedders))
ax.set_xticklabels(['\n'.join(['{}: {}'.format(k, v) for k, v in param2val.items()
                               if k not in INCLUDE_DICT.keys()])
                    for param2val in param2val_list], fontsize=AX_FONTSIZE)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=AX_FONTSIZE)
# legend
labels2 = ['expert', 'shuffled', 'novice']
plt.tight_layout()
add_double_legend(lines, labels1, labels2)
fig.subplots_adjust(bottom=0.1)
plt.show()

# TODO each plot should vary along a singel parameter dimension only (instead of presenting all subtypes of a class)
# TODO then one can stack figures in a powerpoint
# TODO this also simolifies labeling of figures

