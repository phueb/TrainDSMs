import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from sklearn.manifold import TSNE


from two_process_nlp.params import to_embedder_name
from two_process_nlp import config

from analyze.utils import gen_param2vals_for_completed_jobs


LOCAL = True
EMBEDDER_NAMES = ['ww']

TSNE_PP = 30
SV_NUMS = (1, 2)

LINEWIDTH = 2
TICKLABEL_FONT_SIZE = 6
AXLABEL_FONT_SIZE = 10
FIGSIZE = (20, 10)
DPI = 192


def make_probes_acts_2d_fig(mat, num_cats=10, label_probe=False, is_subtitled=True):
    """
    Returns fig showing probe activations in 2D space using SVD & TSNE.
    """
    palette = np.array(sns.color_palette("hls", num_cats))  # TODO num_cats
    # load data
    probes_acts_cats = np.arange(len(mat))  # [model.hub.probe_store.probe_cat_dict[probe] for probe in model.hub.probe_store.types]
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    pcs = np.dot(u, np.diag(s))
    explained_variance = np.var(pcs, axis=0)
    full_variance = np.var(mat, axis=0)
    expl_var_perc = explained_variance / full_variance.sum() * 100
    act_2d_svd = u[:, SV_NUMS]  # this is correct, and gives same results as pca
    acts_2d_tsne = TSNE(perplexity=TSNE_PP).fit_transform(mat)
    # fig
    fig, axarr = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)
    for n, probes_acts_2d in enumerate([act_2d_svd, acts_2d_tsne]):
        # plot
        palette_ids = [0 for _ in probes_acts_cats]  # TODO get correct id
        axarr[n].scatter(probes_acts_2d[:, 0], probes_acts_2d[:, 1], lw=0, s=8, c=palette[palette_ids])
        # axarr[n].axis('off')
        # axarr[n].axis('tight')
        descr_str = ', '.join(['sv {}: var {:2.0f}%'.format(i, expl_var_perc[i]) for i in SV_NUMS])
        if is_subtitled:
            axarr[n].set_title(['SVD ({})'.format(descr_str), 't-SNE'][n], fontsize=AXLABEL_FONT_SIZE)


        # # add the labels for each cat
        # for cat in model.hub.probe_store.cats:
        #     probes_acts_2d_ids = np.where(np.asarray(probes_acts_cats) == cat)[0]
        #     xtext, ytext = np.median(probes_acts_2d[probes_acts_2d_ids, :], axis=0)
        #     txt = axarr[n].text(xtext, ytext, str(cat), fontsize=TICKLABEL_FONT_SIZE,
        #                         color=palette[model.hub.probe_store.cats.index(cat)])
        #     txt.set_path_effects([
        #         PathEffects.Stroke(linewidth=LINEWIDTH, foreground="w"), PathEffects.Normal()])
        # # add the labels for each probe
        # if label_probe:
        #     for probe in model.hub.probe_store.types:
        #         probes_acts_2d_ids = np.where(np.asarray(model.hub.probe_store.types) == probe)[0]
        #         xtext, ytext = np.median(probes_acts_2d[probes_acts_2d_ids, :], axis=0)
        #         txt = axarr[n].text(xtext, ytext, str(probe), fontsize=TICKLABEL_FONT_SIZE)
        #         txt.set_path_effects([
        #             PathEffects.Stroke(linewidth=LINEWIDTH, foreground="w"), PathEffects.Normal()])
    fig.tight_layout()
    return fig


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
        process2_embed_mats = np.load(p)

        # TODO plot animation across eval_steps to preserve axis

        # TODO load category information about row_words

        # fig
        fig1 = make_probes_acts_2d_fig(process2_embed_mats[0])
        plt.show()
        fig2 = make_probes_acts_2d_fig(process2_embed_mats[-1])
        plt.show()
