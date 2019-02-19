import numpy as np
import matplotlib.pyplot as plt

from two_stage_nlp.aggregator import Aggregator
from analyze.utils import to_diff_df
from analyze.utils import to_label


# make diff_df
ag = Aggregator()
df = ag.make_df(load_from_file=True, verbose=True)
diff_df = to_diff_df(df)


# data
task_names = diff_df['task'].unique()
task_name2ys = {}
for n, task_name in enumerate(task_names):
    ys = diff_df[diff_df['task'] == task_name]['diff_score'].values
    task_name2ys[task_name] = ys
sorted_task_names = sorted(task_name2ys.keys(),
                                     key=lambda task_name: np.mean(task_name2ys[task_name]))

# figure
task_name2color = {level: plt.cm.get_cmap('tab10')(n)
                   for n, level in enumerate(task_names)}

fig, ax = plt.subplots(1, figsize=(10, 6), dpi=200)
ax.set_ylim([-0.1, 0.10])
num_x = len(task_names)
x = np.arange(num_x)
ax.set_xticks(x)
ax.set_xticklabels([to_label(task_name).replace('_', '\n') for task_name in sorted_task_names])
ax.set_ylabel('Balanced Accuracy Difference (Classifier - Comparator)')
ax.set_xlabel('Task')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
for n, task_name in enumerate(sorted_task_names):
    ys = task_name2ys[task_name]
    color = task_name2color[task_name]
    ax.bar(x=n,
           height=ys.mean(),
           width=1.0,
           yerr=ys.std(),
           color=color,
           edgecolor='black')

plt.tight_layout()
plt.show()


