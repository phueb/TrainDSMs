import matplotlib.pyplot as plt

from two_process_nlp.aggregator import Aggregator

ARCHITECTURES = ['comparator', 'classifier']

LEG_FONTSIZE = 16
AX_FONTSIZE = 16
FIGSIZE = (12, 8)
DPI = 200


ag = Aggregator()
df = ag.make_df(load_from_file=True, verbose=True)

# exclude
df.drop(df[df['task'] == 'cohyponyms_syntactic'].index, inplace=True)
df.drop(df[df['embedder'] == 'random_normal'].index, inplace=True)

# figure
fig, axarr = plt.subplots(1, len(ARCHITECTURES), figsize=FIGSIZE, dpi=DPI)
plt.title('Tasks & number of epochs')
for ax, arch in zip(axarr, ARCHITECTURES):
    # ax
    ax.set_ylabel('Balanced Accuracy')
    ax.set_xlabel('Tasks')
    ax.set_ylim([0.5, 0.9])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # plot
    df_filtered = df[df['arch'] == arch]
    ax.set_title(arch)
    df_filtered.groupby(['task', 'num_epochs_per_row_word']).mean()['score'].plot(ax=ax, kind='bar')
plt.tight_layout()
plt.show()


