import numpy as np
import matplotlib.pyplot as plt

from two_stage_nlp.aggregator import Aggregator
from analyze.utils import to_label


FACTOR = 'task'


ag = Aggregator()
df = ag.make_df(load_from_file=True, verbose=True)

# clean df
df.drop(df[df['neg_pos_ratio'] == 0.0].index, inplace=True)
df.drop(df[df['task'] == 'cohyponyms_syntactic'].index, inplace=True)
df.drop(df[df['embedder'] == 'random_normal'].index, inplace=True)
df.drop(df[df['arch'] == 'classifier'].index, inplace=True)


# figure
stages = ['novice', 'expert']
factor_levels = df[FACTOR].unique()
level2color = {level: plt.cm.get_cmap('tab10')(n)
               for n, level in enumerate(factor_levels)}

fig, ax = plt.subplots(1, figsize=(8, 6), dpi=200)
plt.title('Interaction between {} and stage'.format(FACTOR))
ax.set_ylim([0.5, 0.90])
num_x = len(factor_levels)
x = np.arange(2)
ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.set_ylabel('Balanced Accuracy')
ax.set_xlabel('Stage')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
for level in factor_levels:
    df_subset = df[df[FACTOR] == level]
    y = []
    for stage in stages:
        score = df_subset[df_subset['stage'] == stage]['score'].mean()
        y.append(score)
    color = level2color[level]
    ax.plot(x, y, label=to_label(level), color=color, zorder=3, linewidth=2)
ax.legend(loc='best', frameon=False)
plt.tight_layout()
plt.show()


