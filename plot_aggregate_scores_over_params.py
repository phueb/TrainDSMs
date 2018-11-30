import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
import copy

from src import config

MIN_NUM_REPS = 3

EMBEDDER_CLASS = 'count'
EMBEDDER_TYPE = None  # can be None

INCLUDE_DICT = {'num_vocab': 4096, 'corpus_name': 'childes-20180319'}
# INCLUDE_DICT.update({'embed_size': 512})
INCLUDE_DICT.update({'reduce_type': ['svd', 200]})
TASK_CLASS = 'classification'

LEG1_Y = 1.2
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
               bbox_to_anchor=(0.5, LEG1_Y - 0.2), ncol=num_bars, frameon=False, fontsize=LEG_FONTSIZE)
    plt.gca().add_artist(leg1)  # order of legend creation matters here


# collect data for plotting
all_embedders_data = []
param2val_list = []
task_name2y_c = {}
for embedder_p in config.Dirs.runs.glob('*'):
    # check embedder_class and type
    with (embedder_p / 'params.yaml').open('r') as f:
        param2val = yaml.load(f)
    key = EMBEDDER_CLASS + '_type'
    try:
        embedder_type = param2val[key]
    except KeyError:
        if 'random_type' in param2val:
            print('Found RandomControl')
            for task_p in embedder_p.glob('*/**'):
                task_name = task_p.name
                ps = [p for p in task_p.glob('scores_*.csv')]
                if len(ps) < MIN_NUM_REPS:
                    continue
                scores = pd.concat([pd.read_csv(p, index_col=False) for p in task_p.glob('scores_*.csv')], axis=1)
                bool_id_c = (scores['shuffled'] == False).values[:, 0]
                distrs_c = scores[bool_id_c]['exp_score'].values
                largest_distr_c = distrs_c[np.argmax(np.mean(distrs_c, axis=1))]
                task_name2y_c[task_name] = largest_distr_c
        continue
    else:
        if EMBEDDER_TYPE is not None and embedder_type != EMBEDDER_TYPE:
            continue
        else:
            print('\nFound {}'.format(embedder_type))
    # exclude
    if not all([param2val[k] == v for k, v in INCLUDE_DICT.items()]):
        print('Excluding {}'.format(embedder_type))
        continue
    # collect data
    embedder_data = []
    for task_p in embedder_p.glob('*/**'):  # only directories excluding model_p
        task_name = task_p.name
        if TASK_CLASS not in task_name:
            continue
        else:
            print(task_name)
        # get score distributions (distributions are over replications)
        ps = [p for p in task_p.glob('scores_*.csv')]
        if len(ps) < MIN_NUM_REPS:
            continue
        scores = pd.concat([pd.read_csv(p, index_col=False) for p in task_p.glob('scores_*.csv')], axis=1)
        bool_id_c = (scores['shuffled'] == False).values[:, 0]
        bool_id_s = (scores['shuffled'] == True).values[:, 0]
        distrs_c = scores[bool_id_c]['exp_score'].values
        distrs_s = scores[bool_id_s]['exp_score'].values
        score_nov = scores['nov_score'].values[0, 0]
        if distrs_c.size == 0 or distrs_s.size == 0:
            continue
        # get distribution with largest mean (over params)
        largest_distr_c = distrs_c[np.argmax(np.mean(distrs_c, axis=1))]  # correct input-output mapping
        largest_distr_s = distrs_s[np.argmax(np.mean(distrs_s, axis=1))]  # shuffled input-output mapping
        assert len(largest_distr_c) == len(largest_distr_s)
        embedder_data.append((task_name, largest_distr_c, largest_distr_s, score_nov))
    if embedder_data:
        all_embedders_data.append(embedder_data)
        param2val_list.append(param2val)

# remove embedders that did not complete all tasks
num_tasks_list = [len(embedder_data) for embedder_data in all_embedders_data]
argmax = np.argmax(num_tasks_list)
max_num_tasks = num_tasks_list[argmax]
task_names = [d[0] for d in all_embedders_data[argmax]]
labels1 = [task_name.replace('_', '\n') for task_name in task_names]
embedders_data = []
for embedder_data in all_embedders_data:
    num_tasks = len(embedder_data)
    if num_tasks == max_num_tasks:
        embedders_data.append(embedder_data)

# fig
print()
colors = plt.cm.get_cmap('tab10')
num_embedders = len(embedders_data)
fig, ax = plt.subplots(figsize=(WIDTH_PER_EMBEDDER * num_embedders, HEIGHT), dpi=DPI)
embedder = EMBEDDER_TYPE or EMBEDDER_CLASS
title = '{} Performance on {} tasks'.format(embedder, TASK_CLASS)
plt.title(title, fontsize=TITLE_FONTSIZE, y=LEG1_Y)
plt.xlabel(INCLUDE_DICT, fontsize=AX_FONTSIZE)
if TASK_CLASS == 'matching':
    ylabel = 'Balanced Accuracy'
    ylims = [0.5, 1]
    yticks = np.arange(0.5, 1 + Y_STEP_SIZE, Y_STEP_SIZE)
elif TASK_CLASS in ['identification', 'classification']:
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
num_bars = 4
lines = []
bw = (1 / (max_num_tasks + 1)) / num_bars
for embedder_id, embedder_data in enumerate(embedders_data):
    num_tasks = len(embedder_data)
    lines = []
    for task_id, (task_name, y_e, y_s, y_n) in enumerate(embedder_data):
        x = embedder_id + (task_id / (num_tasks + 1))
        y_c = task_name2y_c[task_name]
        l1, = ax.bar(x + 0 * bw, y_e.mean(), width=bw, yerr=y_e.std(), color=colors(task_id))
        l2, = ax.bar(x + 1*bw, y_s.mean(), width=bw, yerr=y_s.std(), color=colors(task_id), alpha=0.75)
        l3, = ax.bar(x + 2*bw, y_c.mean(), width=bw, yerr=y_c.std(), color=colors(task_id), alpha=0.5)
        l4, = ax.bar(x + 3*bw, y_n, width=bw, color=colors(task_id), alpha=0.25)
        lines.append((copy.copy(l1), copy.copy(l2), copy.copy(l3), copy.copy(l4)))
# tick labels
ax.set_xticks(np.arange(num_embedders) + (max_num_tasks / 2 * bw * num_bars))
ax.set_xticklabels(['\n'.join(['{}: {}'.format(k, v) for k, v in param2val.items()
                               if k not in INCLUDE_DICT.keys()])
                    for param2val in param2val_list], fontsize=AX_FONTSIZE)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=AX_FONTSIZE)
# legend
labels2 = ['expert', 'expert +\nrandom supervision', 'expert +\nrandom vectors', 'novice']
plt.tight_layout()
add_double_legend(lines, labels1, labels2)
fig.subplots_adjust(bottom=0.1)
plt.show()

