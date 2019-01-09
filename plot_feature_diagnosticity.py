import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pyprind
from cytoolz import itertoolz
from itertools import chain

from src import config
from src.params import CountParams, RNNParams, Word2VecParams, RandomControlParams
from src.params import gen_combinations
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder

CAT_TYPE = 'sem'


def make_feature_diagnosticity_distribution_fig(mat, name):
    chunk_size = 10  # number of unique default colors
    chunk_ids_list = list(itertoolz.partition_all(chunk_size, np.arange(num_cats)))
    num_rows = len(chunk_ids_list)
    fig, axarr = plt.subplots(nrows=num_rows, ncols=1,
                              figsize=(config.Figs.width, 4 * num_rows),
                              dpi=config.Figs.dpi)
    plt.suptitle(name)
    if not isinstance(axarr, np.ndarray):
        axarr = [axarr]
    for ax, chunk_ids in zip(axarr, chunk_ids_list):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False)
        ax.set_xlabel('F1-score')
        ax.set_ylabel('Frequency')
        ax.set_xlim([0, 1])
        # plot histograms
        for cat, row in zip([cats[i] for i in chunk_ids], mat[chunk_ids, :]):
            print('Highest f1-score for "{}" is {:.2f}'.format(cat, np.max(row)))
            ax.hist(row,
                    bins=None,
                    linewidth=2,
                    histtype='step',
                    label=cat)
        ax.legend(loc='best')
    plt.tight_layout()
    return fig


def make_feature_diagnosticity_fig(mat, name):
    fig, ax = plt.subplots(1, figsize=(config.Figs.width, config.Figs.width), dpi=config.Figs.dpi)
    plt.title(name)
    sns.heatmap(mat,
                ax=ax, square=False, annot=False, cbar_kws={"shrink": .5}, cmap='jet', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks(np.arange(num_cats) + 0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels(cats, rotation=0)
    ax.set_xlabel('Embedding Features')
    # colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['0.0', '0.5', '1.0'])
    cbar.set_label('F1 score')
    plt.tight_layout()
    return fig


p = config.Dirs.tasks / '{}_categories'.format(CAT_TYPE) / '{}_{}.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
both = np.loadtxt(p, dtype='str')
np.random.shuffle(both)
probes, probe_cats = both.T
cats = sorted(set(probe_cats))
cat2probes = {cat: [p for p, c in zip(probes, probe_cats) if c == cat]
              for cat in cats}
num_probes = len(probes)
num_cats = len(cats)

embedders = chain(
    (RNNEmbedder(param2id, param2val) for param2id, param2val in gen_combinations(RNNParams)),
    (W2VecEmbedder(param2id, param2val) for param2id, param2val in gen_combinations(Word2VecParams)),
    (CountEmbedder(param2id, param2val) for param2id, param2val in gen_combinations(CountParams)),
    (RandomControlEmbedder(param2id, param2val) for param2id, param2val in gen_combinations(RandomControlParams))
)

for embedder in embedders:
    # feature diagnosticity about category membership
    probe_embed_mat = np.zeros((num_probes, embedder.dim1))
    for n, p in enumerate(probes):
        probe_embed_mat[n] = embedder.w2e[p]
    true_col = [True for p in probes]
    false_col = [False for p in probes]
    feature_diagnosticity_mat = np.zeros((num_cats, embedder.dim1))
    pbar = pyprind.ProgBar(embedder.dim1)
    print('Making feature_diagnosticity_mat...')
    for col_id, col in enumerate(probe_embed_mat.T):
        pbar.update()
        for cat_id, cat in enumerate(cats):
            target_col = [True if p in cat2probes[cat] else False for p in probes]
            last_f1 = 0.0
            for thr in np.linspace(np.min(col), np.max(col), num=config.Figs.num_diagnosticity_steps):
                thr_col = col > thr
                tp = np.sum((thr_col == target_col) & (thr_col == true_col))   # tp
                fp = np.sum((thr_col != target_col) & (thr_col == true_col))   # fp
                fn = np.sum((thr_col != target_col) & (thr_col == false_col))  # fn
                f1 = (2 * tp) / (2 * tp + fp + fn)
                if f1 > last_f1:
                    feature_diagnosticity_mat[cat_id, col_id] = f1
                    last_f1 = f1
    # plot
    fig1 = make_feature_diagnosticity_distribution_fig(feature_diagnosticity_mat, embedder.name),
    fig2 = make_feature_diagnosticity_fig(feature_diagnosticity_mat, embedder.name)
    # TODO save fig