import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import patheffects
from matplotlib.animation import FuncAnimation

from two_process_nlp.params import to_embedder_name
from two_process_nlp import config

from analyze.utils import gen_param2vals_for_completed_jobs

SAVE_ANIMATION = False
LOCAL = True
EMBEDDER_NAMES = ['ww']
LABELED_CAT = 'SIZE'

METHOD = 'pca'
FIT_ID = -1
TSNE_PP = 30
PC_NUMS = (1, 2)

TITLE_Y = 0.9
TITLE_FONTSIZE = 8
SCATTER_SIZE = 6
LABEL_BORDER = 1
LABEL_FONTSIZE = 8
FIGSIZE = (10, 10)
DPI = 192
ANIM_INTERVAL = 500  # ms
EVAL_STEP_INTERVAL = 1


def make_2d_fig(mat, meta_data_df):
    """
    Returns fig showing probe activations. requires that mat shape is [N, 2]
    """
    # load data
    assert len(meta_data_df) == len(mat)
    probes = meta_data_df['probe'].values.tolist()
    probe_cats = meta_data_df['cat'].values.tolist()
    # fig
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    if METHOD == 'pca':
        xlim = 30
        ylim = 30
    elif METHOD == 'tsne':
        xlim = 30
        ylim = 30
    else:
        raise AttributeError('Invalid arg to "METHOD".')
    ax.set(xlim=(-xlim, xlim), ylim=(-ylim, ylim))
    ax.axis('off')
    ax.set_title(title(0), y=TITLE_Y, fontsize=TITLE_FONTSIZE)
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
        if row_id < len(mat) * (1 / config.Eval.num_folds):
            marker = '*'
            size = SCATTER_SIZE + 30
        else:
            marker = 'o'
            size = SCATTER_SIZE
        scatter = ax.scatter(row[0], row[1], marker=marker, lw=0, s=size, c=np.array([color]))
        scatters.append(scatter)
        # label point
        text_str = row_word if LABELED_CAT in probe_cat else ''
        xpos, ypos = mat[row_id, col_ids]
        text = ax.text(xpos + 0.05, ypos, text_str, fontsize=LABEL_FONTSIZE, color=color)
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
        # fit model on last eval_step
        if METHOD == 'tsne':
            fitter = TSNE(perplexity=TSNE_PP).fit(process2_embed_mats[FIT_ID])
            col_ids = [0, 1]
        elif METHOD == 'pca':
            fitter = PCA().fit(process2_embed_mats[FIT_ID])
            col_ids = PC_NUMS
        else:
            raise AttributeError('Invalid arg to "METHODS".')
        # plot initial eval_step

        # TODO multiprocessing: fit_transform in parallel

        mats_2d = [fitter.fit_transform(m) for m in process2_embed_mats]
        title = lambda i: '{} of embedder={} test word vectors\n{}\nFrame {}/{}'.format(
            METHOD, embedder_name, p.relative_to(runs_dir / param_name / job_name), i + 1, len(mats_2d))
        fig, ax, scatters, texts, labels = make_2d_fig(mats_2d[0], meta_data_df)


        def animate(i):
            ax.set_title(title(i),
                y=TITLE_Y, fontsize=TITLE_FONTSIZE)
            for row_id, (scatter, label, text) in enumerate(zip(scatters, labels, texts)):
                offset = mats_2d[i][row_id, col_ids]
                scatter.set_offsets(np.array([offset]))  # offsets must be passed an N×2 array
                xpos, ypos = offset
                text.set_position((xpos + 0.05, ypos))


        anim = FuncAnimation(fig, animate, interval=ANIM_INTERVAL, frames=len(mats_2d))
        #
        if SAVE_ANIMATION:
            anim.save('{}.gif'.format(LABELED_CAT))
        plt.draw()
        plt.show()
