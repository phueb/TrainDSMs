import matplotlib.pyplot as plt
import numpy as np

from two_process_nlp.aggregator import Aggregator

ARCHITECTURES = ['comparator', 'classifier']
EVAL = 'identification'  # changes ylim and ylabel

AX_FONTSIZE = 16
LEG_FONTSIZE = 10
FIGSIZE = (12, 6)
DPI = 200


ag = Aggregator()
df = ag.make_df(load_from_file=True, verbose=True)

# exclude
df.drop(df[df['task'] == 'cohyponyms_syntactic'].index, inplace=True)
df.drop(df[df['embedder'] == 'random_normal'].index, inplace=True)


if EVAL == 'matching':
    ylabel = 'Balanced Accuracy'
    ylims = [0.5, 0.85]
elif EVAL == 'identification':
    ylabel = 'Accuracy'
    ylims = [0.0, 0.8]
else:
    raise AttributeError('Invalid arg to "EVAL".')
embedder_names = ['ww', 'wd', 'sg', 'cbow', 'srn', 'lstm', 'random_normal']
for embedder in embedder_names:
    # include
    df_filtered = df[df['embedder'] == embedder]
    if len(df_filtered) == 0:
        continue

    # figure
    fig, axarr = plt.subplots(1, len(ARCHITECTURES), figsize=FIGSIZE, dpi=DPI)
    plt.suptitle(embedder)
    x = [int(i) for i in df_filtered['num_epochs'].unique() if not np.isnan(i)]
    for ax, arch in zip(axarr, ARCHITECTURES):
        # ax
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
        df_filtered2 = df_filtered[df_filtered['arch'] == arch]
        ax.set_title(arch)
        for task_name, group in df_filtered2.groupby('task'):
            y = group.groupby('num_epochs').mean()['score']
            ax.plot(x, y, label=task_name)
    plt.legend(bbox_to_anchor=(1.0, 0.5), ncol=1,
              frameon=False, fontsize=LEG_FONTSIZE)
    plt.tight_layout()
    plt.show()


