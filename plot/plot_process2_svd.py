import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import patheffects
from matplotlib.animation import FuncAnimation

from two_process_nlp.params import to_embedder_name
from two_process_nlp import config

from analyze.utils import gen_param2vals_for_completed_jobs

METHODS = ['tsne']
LOCAL = True
EMBEDDER_NAMES = ['ww']

TSNE_PP = 30
PC_NUMS = (1, 0)
NUM_ROW_WORDS = 50

SCATTER_SIZE = 8
XLIM = 20
YLIM = 20
LABEL_BORDER = 0
LABEL_FONTSIZE = 4
FIGSIZE = (6, 6)
DPI = 192
ANIM_INTERVAL = 1000
EVAL_STEP_INTERVAL = 2


def make_2d_fig(mat, meta_data_df, num_row_words):
    """
    Returns fig showing probe activations in 2D space using TSNE.
    """
    num_cats = len(meta_data_df['cat'].unique())
    palette = np.array(sns.color_palette("hls", num_cats))
    # load data
    assert len(meta_data_df) == len(mat)
    probe_cats = meta_data_df['cat']
    cats = list(sorted(meta_data_df['cat'].unique()))
    # fig
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set(xlim=(-XLIM, XLIM), ylim=(-YLIM, YLIM))
    # plot
    palette_ids = [cats.index(probe_cat) for probe_cat in probe_cats]
    scat = ax.scatter(mat[:num_row_words, 0], mat[:num_row_words, 1],
                      lw=0, s=SCATTER_SIZE, c=palette[palette_ids][:num_row_words])
    # axarr[n].axis('off')
    # axarr[n].axis('tight')
    # label probes
    for probe in meta_data_df['probe'].values[:num_row_words]:
        probes_acts_2d_ids = np.where(meta_data_df['probe'].values == probe)[0]
        xpos, ypos = np.median(mat[probes_acts_2d_ids, :], axis=0)
        txt = ax.text(xpos + 0.01, ypos + 0.01, str(probe), fontsize=LABEL_FONTSIZE)
        txt.set_path_effects([
            patheffects.Stroke(linewidth=LABEL_BORDER, foreground="w"), patheffects.Normal()])
    fig.tight_layout()
    return fig, scat



embedder_name2plot_data = {embedder_name: [] for embedder_name in EMBEDDER_NAMES}
job_name2plot_data = {}
for param2val in gen_param2vals_for_completed_jobs(local=LOCAL):
    embedder_name = to_embedder_name(param2val)
    param_name = param2val['param_name']
    job_name = param2val['job_name']
    print('\n==================================\nUsing param2val for {}'.format(embedder_name))
    # plot
    runs_dir = config.LocalDirs.runs if LOCAL else config.RemoteDirs.runs
    for p in (runs_dir / param_name / job_name).rglob('process2_embed_mats.npy'):
        # load data
        process2_embed_mats = np.load(p)
        meta_data_df = pd.read_csv(p.parent / 'task_metadata.csv')
        # fig
        make_fig = None
        for method in METHODS:
            # fit model on last eval_step
            if method == 'tsne':
                fitter = TSNE(perplexity=TSNE_PP).fit(process2_embed_mats[-1])
            elif method == 'svd':
                fitter = PCA(n_components=max(PC_NUMS) + 1).fit(process2_embed_mats[-1])
            else:
                raise AttributeError('Invalid arg to "METHODS".')
            # plot initial eval_step
            mats_2d = [fitter.fit_transform(m) for m in process2_embed_mats][::EVAL_STEP_INTERVAL]
            fig, scat = make_2d_fig(mats_2d[0], meta_data_df, num_row_words=NUM_ROW_WORDS)


            def animate(i):
                scat.set_offsets(mats_2d[i])  # offsets must be passed an N×2 array

                # TODO update text

            anim = FuncAnimation(fig, animate, interval=ANIM_INTERVAL, frames=len(mats_2d) - 1)
            plt.draw()
            plt.show()

            # u, s, v = np.linalg.svd(mat, full_matrices=False)
            # pcs = np.dot(u, np.diag(s))
            # explained_variance = np.var(pcs, axis=0)
            # full_variance = np.var(mat, axis=0)
            # expl_var_perc = explained_variance / full_variance.sum() * 100
            # res = u[:, PC_NUMS]  # this is correct, and gives same results as pca



