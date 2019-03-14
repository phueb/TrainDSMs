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

FIT_ID = 0

LOCAL = True
EMBEDDER_NAMES = ['ww']
LABELED_CAT = 'INTELIGENCE'

METHODS = ['pca']
TSNE_PP = 30  # TODO vary
PC_NUMS = (1, 2)  # TODO vary

SCATTER_SIZE = 6
XLIM = 20
YLIM = 20
LABEL_BORDER = 1
LABEL_FONTSIZE = 6
FIGSIZE = (6, 6)
DPI = 192
ANIM_INTERVAL = 1000  # ms
EVAL_STEP_INTERVAL = 1


def make_2d_fig(mat, meta_data_df):
    """
    Returns fig showing probe activations in 2D space using TSNE.
    """
    # num_labeled_cats = len(LABELED_CATS)
    # palette = np.array(sns.color_palette("hls", num_labeled_cats + 1))
    # load data
    assert len(meta_data_df) == len(mat)
    probes = meta_data_df['probe'].values.tolist()
    probe_cats = meta_data_df['cat'].values.tolist()
    # fig
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set(xlim=(-XLIM, XLIM), ylim=(-YLIM, YLIM))
    ax.axis('off')
    # plot
    scatters = []
    texts = []
    labels = []
    for row_id, row in enumerate(mat):
        # plot single point
        row_word = probes[row_id]
        probe_cat = probe_cats[row_id]
        if probe_cat == LABELED_CAT + '+':
            color = 'blue'
        elif probe_cat == LABELED_CAT + '-':
            color = 'red'
        else:
            color = 'black'
        scatter = ax.scatter(row[0], row[1], lw=0, s=SCATTER_SIZE, c=np.array([color]))
        scatters.append(scatter)
        # label point
        text_str = row_word if LABELED_CAT in probe_cat else ''
        xpos, ypos = mat[row_id, PC_NUMS]
        text = ax.text(xpos + 0.01, ypos + 0.01, text_str, fontsize=LABEL_FONTSIZE, color=color)
        text.set_path_effects([
            patheffects.Stroke(linewidth=LABEL_BORDER, foreground="w"), patheffects.Normal()])
        texts.append(text)
        labels.append(row_word)
    fig.tight_layout()
    return fig, ax, scatters, texts, labels


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
        process2_embed_mats = np.load(p)[::EVAL_STEP_INTERVAL]
        meta_data_df = pd.read_csv(p.parent / 'task_metadata.csv')
        print(meta_data_df['cat'].unique())
        # fig
        make_fig = None
        for method in METHODS:
            # fit model on last eval_step
            if method == 'tsne':
                fitter = TSNE(perplexity=TSNE_PP).fit(process2_embed_mats[FIT_ID])
            elif method == 'pca':
                fitter = PCA().fit(process2_embed_mats[FIT_ID])
            else:
                raise AttributeError('Invalid arg to "METHODS".')
            # plot initial eval_step

            # TODO multiprocessing: fit_transform in parallel

            mats_2d = [fitter.fit_transform(m) for m in process2_embed_mats]
            fig, ax, scatters, texts, labels = make_2d_fig(mats_2d[0], meta_data_df)

            def animate(i):
                ax.set_title('Frame {}/{}'.format(i + 1, len(mats_2d)))
                for row_id, (scatter, label, text) in enumerate(zip(scatters, labels, texts)):
                    offset = mats_2d[i][row_id, PC_NUMS]
                    scatter.set_offsets(np.array([offset]))  # offsets must be passed an N×2 array
                    xpos, ypos = offset
                    text.set_position((xpos + 0.01, ypos + 0.01))

            anim = FuncAnimation(fig, animate, interval=ANIM_INTERVAL, frames=len(mats_2d))
            plt.draw()
            plt.show()

            # u, s, v = np.linalg.svd(mat, full_matrices=False)
            # pcs = np.dot(u, np.diag(s))
            # explained_variance = np.var(pcs, axis=0)
            # full_variance = np.var(mat, axis=0)
            # expl_var_perc = explained_variance / full_variance.sum() * 100
            # res = u[:, PC_NUMS]  # this is correct, and gives same results as pca



