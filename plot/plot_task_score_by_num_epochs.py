import matplotlib.pyplot as plt
import numpy as np

from two_process_nlp.aggregator import Aggregator

ARCHITECTURES = ['comparator', 'classifier']

AX_FONTSIZE = 16
LEG_FONTSIZE = 10
FIGSIZE = (10, 4)
DPI = 200


ag = Aggregator()
df = ag.make_df(load_from_file=False, verbose=True)

# exclude
df.drop(df[df['task'] == 'cohyponyms_syntactic'].index, inplace=True)
df.drop(df[df['embedder'] == 'random_normal'].index, inplace=True)

# figure
fig, axarr = plt.subplots(1, len(ARCHITECTURES), figsize=FIGSIZE, dpi=DPI)
plt.title('Tasks & number of epochs')
x = [i for i in df['num_epochs'].unique() if not np.isnan(i)]
for ax, arch in zip(axarr, ARCHITECTURES):
    # ax
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_xlabel('num_epochs')
    ax.set_ylim([0.5, 0.85])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # plot
    df_filtered = df[df['arch'] == arch]
    ax.set_title(arch)
    for task_name, group in df_filtered.groupby('task'):
        print(task_name)

        # TODO debug
        # print(group.groupby('num_epochs').mean())

        y = group.groupby('num_epochs').mean()['score']
        ax.plot(x, y, label=task_name)
plt.legend(bbox_to_anchor=(1.0, 0.5), ncol=1,
          frameon=False, fontsize=LEG_FONTSIZE)
plt.tight_layout()
plt.show()


