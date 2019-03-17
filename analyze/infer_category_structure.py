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

FIGSIZE = (16, 16)
DPI = None
AXLABEL_FONTSIZE = 12


def make_category_structure_fig(task_name, mat, original_ytick_labels, original_xtick_labels):
    print('Preparing clustermap...')
    fig, ax_heatmap = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    title = 'Category Structure of\n{}'.format(task_name)
    divider = make_axes_locatable(ax_heatmap)
    ax_dendleft = divider.append_axes("right", 1.0, pad=0.0, sharey=ax_heatmap)
    ax_dendleft.set_frame_on(False)
    ax_dendtop = divider.append_axes("top", 1.0, pad=0.0, sharex=ax_heatmap)
    ax_dendtop.set_frame_on(False)
    # side dendrogram
    print('Clustering along rows...')
    lnk0 = linkage(pdist(mat))
    dg0 = dendrogram(lnk0,
                     ax=ax_dendleft,
                     orientation='right',
                     color_threshold=-1,
                     no_labels=True)
    z = mat[dg0['leaves'], :]  # reorder rows
    z = z[::-1]  # reverse to match orientation of dendrogram
    # top dendrogram
    print('Clustering along columns...')
    lnk1 = linkage(pdist(mat.T))
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
    ax_heatmap.imshow(z,
                      aspect='equal',
                      cmap=plt.cm.jet,
                      interpolation='nearest',
                      extent=(0, max_x_extent, 0, max_y_extent))
    # label axes
    ax_heatmap.set_ylabel('Probes', fontsize=AXLABEL_FONTSIZE)
    ax_heatmap.set_xlabel('Relata', fontsize=AXLABEL_FONTSIZE)
    # xticks
    num_cols = len(mat.T)
    assert num_cols == mat.shape[1]
    halfxw = 0.5 * max_x_extent / num_cols
    ax_heatmap.set_xticks([])  # np.linspace(halfxw, max_x_extent - halfxw, num_cols)
    xtick_labels = []  #np.array(original_xtick_labels)[dg1['leaves']][::-1]
    ax_heatmap.xaxis.set_ticklabels(xtick_labels, rotation=90, fontsize=18)
    # yticks
    num_rows = len(mat)
    assert num_rows == mat.shape[0]
    halfyw = 0.5 * max_y_extent / num_rows
    ax_heatmap.set_yticks([])  # np.linspace(halfyw, max_y_extent - halfyw, num_rows)
    ytick_labels = []  # np.array(original_ytick_labels)[dg0['leaves']][::-1]
    ax_heatmap.yaxis.set_ticklabels(ytick_labels, rotation=0)
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
    return fig


for param2val in gen_param2vals_for_completed_jobs(LOCAL):
    # embedder
    embedder_name = to_embedder_name(param2val)
    job_name = param2val['job_name']
    print('\n==================================\nUsing param2val for {}'.format(embedder_name))
    embedder = init_embedder(param2val)
    embedder.load_w2e(LOCAL)
    #
    for ev in [
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
                    if col_word in candidates and col_word in ev.probe2relata[row_word]:
                        gold_mat[i, j] = 1
        elif ev.name == 'matching':
            raise NotImplementedError
        else:
            raise AttributeError('Invalid arg to "name".')

        print('Shape of gold_mat={}'.format(gold_mat.shape))
        print(np.sum(gold_mat, axis=1))
        print(np.var(np.sum(gold_mat, axis=1)))
        print(np.sum(gold_mat, axis=0))
        print(np.var(np.sum(gold_mat, axis=0)))
        print()
        # fig
        fig = make_category_structure_fig(ev.full_name, gold_mat, ev.row_words, ev.col_words)
    plt.show()