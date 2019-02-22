import numpy as np
import matplotlib.pyplot as plt

from two_stage_nlp.aggregator import Aggregator
from analyze.utils import to_label


FACTOR = 'task'
STAGES = ['novice', 'expert']

LEG_FONTSIZE = 16
AX_FONTSIZE = 16
FIGSIZE = (8, 6)
DPI = 200


ag = Aggregator()
df = ag.make_df(load_from_file=True, verbose=True)

# clean df
df.drop(df[df['neg_pos_ratio'] == 0.0].index, inplace=True)
df.drop(df[df['task'] == 'cohyponyms_syntactic'].index, inplace=True)
df.drop(df[df['embedder'] == 'random_normal'].index, inplace=True)
df.drop(df[df['arch'] == 'comparator'].index, inplace=True)


# figure

factor_levels = df[FACTOR].unique().tolist()
level2color = {level: plt.cm.get_cmap('tab10')(n)
               for n, level in enumerate(factor_levels)}

fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
if FACTOR == 'embedder':
    factor = 'stage-1-model'
elif FACTOR == 'arch':
    factor = 'stage-2-model'
else:
    factor = FACTOR
plt.title('Interaction between {} and stage'.format(factor), fontsize=AX_FONTSIZE)
ax.set_ylim([0.5, 0.90])
num_x = len(factor_levels)
x = np.arange(2)
ax.set_xticks(x)
ax.set_xticklabels(STAGES, fontsize=AX_FONTSIZE)
ax.set_ylabel('Balanced Accuracy', fontsize=AX_FONTSIZE)
ax.set_xlabel('Stage', fontsize=AX_FONTSIZE)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
for level in factor_levels:
    df_subset = df[df[FACTOR] == level]
    y = []
    print(level)
    for stage in STAGES:
        score = df_subset[df_subset['stage'] == stage]['score'].mean()
        print(stage, score)
        y.append(score)
    color = level2color[level]
    ax.plot(x, y, label=to_label(level), color=color, zorder=3, linewidth=2)
ax.legend(loc='best', frameon=False, fontsize=LEG_FONTSIZE,
          bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.show()


