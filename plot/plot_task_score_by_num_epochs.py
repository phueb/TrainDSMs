import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable

from two_process_nlp.aggregator import Aggregator

ARCHITECTURES = ['extractor', 'comparator', 'aligner', 'classifier']
EVAL = 'identification'  # changes ylim and ylabel
CHANCE = 0.25

LOAD_FROM_FILE = True

AX_FONTSIZE = 16
LEG_FONTSIZE = 10
FIGSIZE = (16, 6)
DPI = 200


ag = Aggregator()
df = ag.make_df(load_from_file=LOAD_FROM_FILE, verbose=True)

# exclude
df.drop(df[df['task'] == 'cohyponyms_syntactic'].index, inplace=True)
df.drop(df[df['standardize'] == 0].index, inplace=True)

# eval
if EVAL == 'matching':
    ylabel = 'Balanced Accuracy'
    ylims = [0.5, 0.85]
elif EVAL == 'identification':
    ylabel = 'Accuracy'
    ylims = [0.0, 0.8]
else:
    raise AttributeError('Invalid arg to "EVAL".')


embedder_names = ['random_normal']  #, 'wd', 'sg', 'cbow', 'srn', 'lstm', 'random_normal']
task_names = df['task'].unique()
task_name2color = {t: plt.cm.get_cmap('tab10')(n) for n, t in enumerate(task_names)}
for embedder in embedder_names:
    # filter
    df_filtered = df[df['embedder'] == embedder]
    if len(df_filtered) == 0:
        continue

    # figure
    fig, axarr = plt.subplots(1, len(ARCHITECTURES), figsize=FIGSIZE, dpi=DPI)
    plt.suptitle('embedder={}'.format(embedder))
    x = [int(i) for i in df_filtered['num_epochs'].unique() if not np.isnan(i)]
    for axis_id, (ax, arch) in enumerate(zip(axarr, ARCHITECTURES)):
        # ax
        make_axes_area_auto_adjustable(ax, pad=0.4)
        ax.set_xticks([x[0], x[-1]])
        ax.set_xticklabels([x[0], x[-1]])
        ax.set_ylabel(ylabel)
        ax.set_xlabel('num_epochs')
        ax.yaxis.grid(True)
        ax.set_ylim(ylims)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False)
        # plot
        ax.axhline(y=CHANCE, color='black')
        df_filtered2 = df_filtered[df_filtered['arch'] == arch]
        ax.set_title(arch)
        for task_name, group in df_filtered2.groupby('task'):
            color = task_name2color[task_name]
            label = task_name if axis_id == 0 else '_nolegend_'
            #
            means = group.groupby('num_epochs').mean()['score']
            stds = group.groupby('num_epochs').std()['score']
            ax.plot(x, means, label=label, color=color)
            ax.fill_between(x, means - stds / 2, means + stds / 2, facecolor=color, alpha=0.3)
            # console
            num_reps = group.groupby('num_epochs').size().mean()
            print(task_name)
            print('num reps={}\n'.format(num_reps))
    fig.legend(bbox_to_anchor=(0.5, 0.9), ncol=len(task_names),
               frameon=False, fontsize=LEG_FONTSIZE, loc='lower center')
    plt.tight_layout()
    plt.show()


