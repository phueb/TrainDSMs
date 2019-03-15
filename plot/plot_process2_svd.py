import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import patheffects
from matplotlib.animation import FuncAnimation

from two_process_nlp.params import to_embedder_name
from two_process_nlp import config

from analyze.utils import gen_param2vals_for_completed_jobs

SAVE_ANIMATION = True
LOCAL = True
EMBEDDER_NAMES = ['ww']
LABELED_CAT = 'FILLED'

METHOD = 'pca'
FIT_ID = 0
TSNE_PP = 30
PC_NUMS = (1, 2)

TITLE_Y = 0.9
TITLE_FONTSIZE = 8
SCATTER_SIZE = 6
LABEL_BORDER = 1
LABEL_FONTSIZE = 8
FIGSIZE = (8, 8)
DPI = 300
ANIM_INTERVAL = 1000  # ms
EVAL_STEP_INTERVAL = 1


def make_2d_fig(mat, meta_data):
    """
    Returns fig showing probe activations. requires that mat shape is [N, 2]
    """
    assert len(meta_data) == len(mat)
    # fig
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    if METHOD == 'pca':
        xlim = 20
        ylim = 20
    elif METHOD == 'tsne':
        xlim = 30
        ylim = 30
    else:
        raise AttributeError('Invalid arg to "METHOD".')
    ax.set(xlim=(-xlim, xlim), ylim=(-ylim, ylim))
    ax.axis('off')
    ax.set_title(title(0), y=TITLE_Y, fontsize=TITLE_FONTSIZE)

    def are_cats_secondary(p_cats, s_cats):
        for p_cat in p_cats:
            if p_cat in s_cats:
                return True
        else:
            return False

    # plot
    scatters = []
    texts = []
    labels = []
    secondary_cats_pos = set()
    secondary_cats_neg = set()
    for row_id, (row, meta_d) in enumerate(zip(mat, meta_data)):
        # color
        probe, probe_cats = meta_d
        if LABELED_CAT + '+' in probe_cats:
            color = 'blue'
            text_str = probe
            print('+', probe, probe_cats)
            secondary_cats_pos.update(probe_cats)
        elif LABELED_CAT + '-' in probe_cats:
            color = 'red'
            text_str = probe
            print('-', probe, probe_cats)
            secondary_cats_neg.update(probe_cats)
        elif are_cats_secondary(probe_cats, secondary_cats_pos):  # probe shares secondary cat with + probe
            color = 'green'
            text_str = probe
            print('+', probe, probe_cats)
        elif are_cats_secondary(probe_cats, secondary_cats_neg):  # probe shares secondary cat with - probe
            color = 'orange'
            text_str = probe
            print('-', probe, probe_cats)
        else:
            color = 'black'
            text_str = ''
        # marker
        if row_id < len(mat) * (1 / config.Eval.num_folds):
            marker = '*'
            size = SCATTER_SIZE + 30
        else:
            marker = 'o'
            size = SCATTER_SIZE
        # plot single point
        scatter = ax.scatter(row[0], row[1], marker=marker, lw=0, s=size, c=np.array([color]))
        scatters.append(scatter)
        # label point
        xpos, ypos = mat[row_id, col_ids]
        text = ax.text(xpos + 0.05, ypos, text_str, fontsize=LABEL_FONTSIZE, color=color)
        text.set_path_effects([
            patheffects.Stroke(linewidth=LABEL_BORDER, foreground="w"), patheffects.Normal()])
        texts.append(text)
        labels.append(probe)
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
        # load data + meta data
        process2_embed_mats = np.load(p)[::EVAL_STEP_INTERVAL]
        with (p.parent / 'task_metadata.pkl').open('rb') as f:
            meta_data = pickle.load(f)  # is a list like [(row_word, cats), ...]
        # console
        print(set([i for md in meta_data for i in md[1]]))
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
        title = lambda i: '{} of embedder={} vector space\n{}\ntest=o train=*\n{}\nFrame {}/{}'.format(
            METHOD, embedder_name, LABELED_CAT, p.relative_to(runs_dir / param_name / job_name), i + 1, len(mats_2d))
        fig, ax, scatters, texts, labels = make_2d_fig(mats_2d[0], meta_data)


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
            anim.save('{}.gif'.format(LABELED_CAT), writer='imagemagick')
        plt.draw()
        plt.show()
