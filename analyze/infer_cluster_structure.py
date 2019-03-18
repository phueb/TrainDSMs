import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import pdist
import numpy as np

from two_process_nlp import config
from two_process_nlp.architectures import comparator
from two_process_nlp.evaluators.matching import Matching
from two_process_nlp.evaluators.identification import Identification
from two_process_nlp.utils import init_embedder
from two_process_nlp.params import to_embedder_name

from analyze.utils import gen_param2vals_for_completed_jobs

LOCAL = True

CORPUS_NAME = config.Corpus.name
NUM_VOCAB = config.Corpus.num_vocab

INCLUDE_ALL_RELATA = True  # if False, use only positive relations that are in eval_candidates_mat

PLOT_NUM_ROWS = None
FIGSIZE = (10, 16)
DPI = None
TICKLABEL_FONTSIZE = 20
AXLABEL_FONTSIZE = 20


def make_cluster_structure_fig(task_name, mat, original_row_words, original_col_words):
    print('Preparing clustermap...')
    fig, ax_heatmap = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    title = 'Cluster Structure of\n{}'.format(task_name)
    divider = make_axes_locatable(ax_heatmap)
    ax_dendleft = divider.append_axes("right", 1.0, pad=0.0, sharey=ax_heatmap)
    ax_dendleft.set_frame_on(False)
    ax_dendtop = divider.append_axes("top", 1.0, pad=0.0, sharex=ax_heatmap)
    ax_dendtop.set_frame_on(False)
    # side dendrogram
    print('Clustering along rows...')
    lnk0 = linkage(pdist(mat), metric='hamming')
    dg0 = dendrogram(lnk0,
                     ax=ax_dendleft,
                     orientation='right',
                     color_threshold=-1,
                     no_labels=True,
                     no_plot=True if PLOT_NUM_ROWS is not None else False)
    z = mat[dg0['leaves'], :]  # reorder rows
    z = z[::-1]  # reverse to match orientation of dendrogram
    # top dendrogram
    print('Clustering along columns...')
    lnk1 = linkage(pdist(mat.T), metric='hamming')
    dg1 = dendrogram(lnk1,
                     ax=ax_dendtop,
                     color_threshold=-1,
                     no_labels=True)
    z = z[:, dg1['leaves']]  # reorder cols to match leaves of dendrogram
    z = z[::-1]  # reverse to match orientation of dendrogram
    # heatmap
    print('Plotting heatmap...')
    max_x_extent = ax_dendtop.get_xlim()[1]
    max_y_extent = ax_dendleft.get_ylim()[1]
    ax_heatmap.imshow(z[:PLOT_NUM_ROWS],
                      aspect='auto' if PLOT_NUM_ROWS else 'equal',
                      cmap=plt.get_cmap('jet'),
                      interpolation='nearest',
                      extent=(0, max_x_extent, 0, max_y_extent))
    # reordered labels
    col_labels = np.array(original_col_words)[dg1['leaves']]
    row_labels = np.array(original_row_words)[dg0['leaves']]
    # label axes
    ax_heatmap.set_ylabel('Probes', fontsize=AXLABEL_FONTSIZE)
    ax_heatmap.set_xlabel('Relata', fontsize=AXLABEL_FONTSIZE)
    # xticks
    num_cols = len(mat.T)
    halfxw = 0.5 * max_x_extent / num_cols
    xticks = np.linspace(halfxw, max_x_extent - halfxw, num_cols) if PLOT_NUM_ROWS else []
    ax_heatmap.set_xticks(xticks)
    xtick_labels = col_labels if PLOT_NUM_ROWS else []  # no need to reverse col_labels
    ax_heatmap.xaxis.set_ticklabels(xtick_labels, rotation=90, fontsize=TICKLABEL_FONTSIZE)
    # yticks
    num_rows = PLOT_NUM_ROWS or len(mat)
    halfyw = 0.5 * max_y_extent / num_rows
    yticks = np.linspace(halfyw, max_y_extent - halfyw, num_rows) if PLOT_NUM_ROWS else []
    ax_heatmap.set_yticks(yticks)
    ytick_labels = row_labels[:PLOT_NUM_ROWS][::-1] if PLOT_NUM_ROWS else []  # only reverse row_labels
    ax_heatmap.yaxis.set_ticklabels(ytick_labels, rotation=0, fontsize=TICKLABEL_FONTSIZE)
    # title
    plt.title(title, fontsize=30)
    # remove dendrogram ticklines
    lines = (ax_dendtop.xaxis.get_ticklines() +
             ax_dendtop.yaxis.get_ticklines() +
             ax_dendleft.xaxis.get_ticklines() +
             ax_dendleft.yaxis.get_ticklines())
    plt.setp(lines, visible=False)
    # make dendrogram labels invisible
    plt.setp(ax_dendleft.get_yticklabels() + ax_dendleft.get_xticklabels(),
             visible=False)
    plt.setp(ax_dendtop.get_xticklabels() + ax_dendtop.get_yticklabels(),
             visible=False)
    fig.tight_layout()
    return z, row_labels, col_labels


def calc_num_clusters(vec):
    where_zeros = np.equal(vec.astype(bool), False)[:-1]
    where_one_or_zero_starts = np.equal(~np.not_equal(vec[:-1], vec[1:]), False)

    cluster_starts = where_zeros & where_one_or_zero_starts
    num_clusters = np.sum(cluster_starts)
    # can't detect 1 in first position
    if np.count_nonzero(vec[0]) == 1:
        num_clusters += 1
    return num_clusters


for param2val in gen_param2vals_for_completed_jobs(LOCAL):
    # embedder
    embedder_name = to_embedder_name(param2val)
    job_name = param2val['job_name']
    print('\n==================================\nUsing param2val for {}'.format(embedder_name))
    embedder = init_embedder(param2val)
    embedder.load_w2e(LOCAL)
    #
    for ev in [

        # Identification(comparator, 'nyms', 'syn', suffix='_test'),
        # Identification(comparator, 'nyms', 'ant', suffix='_test'),

        # Identification(comparator, 'nyms', 'syn', suffix='_jwunique'),
        # Identification(comparator, 'nyms', 'ant', suffix='_jwunique'),

        Identification(comparator, 'nyms', 'syn', suffix='_jw'),
        Identification(comparator, 'nyms', 'ant', suffix='_jw'),
    ]:
        # make eval data - row_words can contain duplicates
        vocab_sims_mat = None
        all_eval_probes, all_eval_candidates_mat = ev.make_all_eval_data(vocab_sims_mat, embedder.vocab)
        ev.row_words, ev.col_words, ev.eval_candidates_mat = ev.downsample(
            all_eval_probes, all_eval_candidates_mat)
        # make gold
        if ev.name == 'identification':
            num_rows = len(ev.row_words)
            num_cols = len(ev.col_words)
            gold_mat = np.zeros((num_rows, num_cols))
            for i in range(num_rows):
                candidates = ev.eval_candidates_mat[i]
                row_word = ev.row_words[i]
                for j in range(num_cols):
                    col_word = ev.col_words[j]
                    if INCLUDE_ALL_RELATA:
                        if 'syn' in ev.full_name and row_word == col_word:
                            gold_mat[i, j] = 1
                        elif 'ant' in ev.full_name and row_word == col_word:
                            gold_mat[i, j] = -1
                        if col_word in ev.probe2relata[row_word]:
                            gold_mat[i, j] = 1
                        elif col_word in ev.probe2lures[row_word]:
                            gold_mat[i, j] = -1
                    elif col_word in candidates:
                        if 'syn' in ev.full_name and row_word == col_word:
                            gold_mat[i, j] = 1
                        elif 'ant' in ev.full_name and row_word == col_word:
                            gold_mat[i, j] = -1
                        if col_word in ev.probe2relata[row_word]:
                            gold_mat[i, j] = 1
                        elif col_word in ev.probe2lures[row_word]:
                            gold_mat[i, j] = -1
        elif ev.name == 'matching':
            raise NotImplementedError
        else:
            raise AttributeError('Invalid arg to "name".')
        print('Shape of gold_mat={}'.format(gold_mat.shape))
        print(np.sum(gold_mat))
        print(np.var(np.sum(gold_mat, axis=1)))
        print(np.var(np.sum(gold_mat, axis=0)))
        print()
        # cluster gold
        # clusters are NOT categories, because a category can be split up into multiple clusters
        # because some category members ocur in multiple clusters
        cluster_mat, row_words, col_words = make_cluster_structure_fig(
            ev.full_name, gold_mat, ev.row_words, ev.col_words)

        # TODO count number of within and between cluster positive and negative links
        # TODO this can only be done with traversing the graph - to distinguish intra vs between cluster relations

        num_row_clusters = 0
        for row, row_word in zip(cluster_mat, row_words):
            num_row_clusters += calc_num_clusters(row)
            # print(row_word, num_clusters, ev.probe2cats[row_word], np.array(col_words)[np.nonzero(row)])
            # print(np.nonzero(row))
        print('num row cluster={}'.format(num_row_clusters))

        num_col_clusters = 0
        for col in cluster_mat.T:
            num_col_clusters += calc_num_clusters(col)
        print('num col cluster={}'.format(num_col_clusters))


        plt.show()

        # raise SystemExit