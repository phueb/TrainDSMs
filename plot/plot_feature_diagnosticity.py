import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import pdist
import numpy as np
import pyprind
from cytoolz import itertoolz
import sys

from two_process_nlp import config
from two_process_nlp.utils import init_embedder

from analyze.utils import gen_param2vals_for_completed_jobs


LOAD_FROM_DISK = True
VERBOSE = False

NUM_DIAGNOSTICITY_STEPS = 100

LEG_FONTSIZE = 16
AX_FONTSIZE = 16
FIGSIZE = (8, 8)
DPI = 100


def make_feature_diagnosticity_distribution_fig(mat, name):
    chunk_size = 10  # number of unique default colors
    chunk_ids_list = list(itertoolz.partition_all(chunk_size, np.arange(num_cats)))
    num_rows = len(chunk_ids_list)
    fig, axarr = plt.subplots(nrows=num_rows, ncols=1,
                              figsize=FIGSIZE,
                              dpi=DPI)
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
            if VERBOSE:
                print('Highest f1-score for "{}" is {:.2f}'.format(cat, np.max(row)))
            ax.hist(row,
                    bins=None,
                    linewidth=2,
                    histtype='step',
                    label=cat)
        ax.legend(loc='best')
    plt.tight_layout()
    return fig


def make_feature_diagnosticity_fig(mat_raw, name):
    # subtract row-mean from elements in each row - large categories have larger overall f1
    mat = mat_raw - mat_raw.mean(axis=1, keepdims=True)
    #
    fig, ax_heatmap = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
    plt.title(name)
    # divide axis
    divider = make_axes_locatable(ax_heatmap)
    ax_dendleft = divider.append_axes("right", 1.0, pad=0.0, sharey=ax_heatmap)
    ax_dendleft.set_frame_on(False)
    ax_colorbar = divider.append_axes("right", 0.2, pad=0.1)
    # side dendrogram
    lnk0 = linkage(pdist(mat))
    dg0 = dendrogram(lnk0,
                     ax=ax_dendleft,
                     orientation='right',
                     color_threshold=-1,
                     no_labels=True)
    z = mat[dg0['leaves'], :]  # reorder rows
    z = z[::-1]  # reverse to match orientation of dendrogram
    # heatmap
    vmin, vmid, vmax = 0.0, 0.5, 1.0
    max_x_extent = ax_heatmap.get_xlim()[1]
    max_y_extent = ax_dendleft.get_ylim()[1]
    im = ax_heatmap.imshow(z,
                           vmin=vmin,
                           vmax=vmax - mat_raw.mean(),
                           aspect='auto',
                           cmap=plt.cm.jet,
                           interpolation='nearest',
                           extent=(0, max_x_extent, 0, max_y_extent))
    # x ticks
    ax_heatmap.set_xticklabels([], fontsize=AX_FONTSIZE)
    ax_heatmap.set_xlabel('Embedding Features', fontsize=AX_FONTSIZE)
    # y ticks
    ylim = ax_heatmap.get_ylim()[1]
    num_cats = len(cats)
    halfyw = 0.5 * ylim / num_cats
    ax_heatmap.yaxis.set_ticks(np.linspace(halfyw, ylim - halfyw, num_cats))
    ax_heatmap.yaxis.set_ticklabels(np.array(cats)[dg0['leaves']], fontsize=AX_FONTSIZE)
    # Hide all tick lines
    lines = (ax_heatmap.xaxis.get_ticklines() +
             ax_heatmap.yaxis.get_ticklines() +
             ax_dendleft.xaxis.get_ticklines() +
             ax_dendleft.yaxis.get_ticklines())
    plt.setp(lines, visible=False)
    # make dendrogram labels invisible
    plt.setp(ax_dendleft.get_yticklabels() + ax_dendleft.get_xticklabels(),
             visible=False)
    # fig.subplots_adjust(bottom=0.2)  # make room for tick labels
    fig.tight_layout()
    # colorbar
    cbar = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax], orientation='vertical')
    cbar.set_ticks([vmin, vmid, vmax])
    cbar.set_ticklabels([str(vmin), str(vmid), str(vmax)])
    cbar.set_label('F1 score (centered row-wise)')
    plt.tight_layout()
    return fig


def make_feature_diagnosticity_mat():
    res = np.zeros((num_cats, embedder.dim1))
    pbar = pyprind.ProgBar(embedder.dim1, stream=sys.stdout)
    print('Making feature_diagnosticity_mat...')
    for col_id, col in enumerate(probe_embed_mat.T):
        pbar.update()
        for cat_id, cat in enumerate(cats):
            target_col = [True if p in cat2probes[cat] else False for p in probes]
            last_f1 = 0.0
            for thr in np.linspace(np.min(col), np.max(col), num=NUM_DIAGNOSTICITY_STEPS):
                thr_col = col > thr
                tp = np.sum((thr_col == target_col) & (thr_col == true_col))  # tp
                fp = np.sum((thr_col != target_col) & (thr_col == true_col))  # fp
                fn = np.sum((thr_col != target_col) & (thr_col == false_col))  # fn
                f1 = (2 * tp) / (2 * tp + fp + fn)
                if f1 > last_f1:
                    res[cat_id, col_id] = f1
                    last_f1 = f1
    return res


p = config.LocalDirs.tasks / 'hypernyms' / '{}_{}.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
both = np.loadtxt(p, dtype='str')
np.random.shuffle(both)
probes, probe_cats = both.T
cats = sorted(set(probe_cats))
cat2probes = {cat: [p for p, c in zip(probes, probe_cats) if c == cat]
              for cat in cats}
num_probes = len(probes)
num_cats = len(cats)

embedder_names = set()
for param2val in gen_param2vals_for_completed_jobs():
    embedder = init_embedder(param2val)
    if embedder.name in embedder_names:
        continue
    else:
        print(embedder.name)
    embedder.load_w2e()
    #
    probe_embed_mat = np.zeros((num_probes, embedder.dim1))
    for n, p in enumerate(probes):
        probe_embed_mat[n] = embedder.w2e[p]
    true_col = [True for p in probes]
    false_col = [False for p in probes]
    #
    file_name = 'feat_diag_mat_{}.npy'.format(embedder.name)
    if LOAD_FROM_DISK:
        try:
            feature_diagnosticity_mat = np.load(file_name)
        except FileNotFoundError:
            feature_diagnosticity_mat = make_feature_diagnosticity_mat()
    else:
        feature_diagnosticity_mat = make_feature_diagnosticity_mat()
    np.save(file_name, feature_diagnosticity_mat)
    # fig
    fig1 = make_feature_diagnosticity_distribution_fig(feature_diagnosticity_mat, embedder.name),
    fig2 = make_feature_diagnosticity_fig(feature_diagnosticity_mat, embedder.name)
    # collect
    embedder_names.add(embedder.name)


plt.show()