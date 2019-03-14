import numpy as np
import matplotlib.pyplot as plt

from two_process_nlp.job_utils import init_embedder
from two_process_nlp.job_utils import w2e_to_sims
from two_process_nlp.params import to_embedder_name

from analyze.utils import gen_param2vals_for_completed_jobs


LOCAL = True
EMBEDDER_NAMES = ['ww']  #, 'wd', 'sg', 'cbow', 'srn', 'lstm', 'random_normal', 'random_uniform']


def make_probes_acts_2d_fig(sv_nums=(1, 2),
                            perplexity=30,
                            label_probe=False,
                            is_subtitled=True):
    """
    Returns fig showing probe activations in 2D space using SVD & TSNE.
    """
    palette = np.array(sns.color_palette("hls", model.hub.probe_store.num_cats))
    # load data
    probes_acts_df = model.get_multi_probe_prototype_acts_df()
    probes_acts_cats = [model.hub.probe_store.probe_cat_dict[probe] for probe in model.hub.probe_store.types]
    u, s, v = linalg.svd(probes_acts_df.values, full_matrices=False)
    pcs = np.dot(u, np.diag(s))
    explained_variance = np.var(pcs, axis=0)
    full_variance = np.var(probes_acts_df.values, axis=0)
    expl_var_perc = explained_variance / full_variance.sum() * 100
    act_2d_svd = u[:, sv_nums]  # this is correct, and gives same results as pca
    acts_2d_tsne = TSNE(perplexity=perplexity).fit_transform(probes_acts_df.values)
    # fig
    fig, axarr = plt.subplots(2, 1, figsize=(FigsConfigs.MAX_FIG_WIDTH, 14), dpi=FigsConfigs.DPI)
    for n, probes_acts_2d in enumerate([act_2d_svd, acts_2d_tsne]):
        # plot
        palette_ids = [model.hub.probe_store.cats.index(probes_acts_cat) for probes_acts_cat in probes_acts_cats]
        axarr[n].scatter(probes_acts_2d[:, 0], probes_acts_2d[:, 1], lw=0, s=8, c=palette[palette_ids])
        axarr[n].axis('off')
        axarr[n].axis('tight')
        descr_str = ', '.join(['sv {}: var {:2.0f}%'.format(i, expl_var_perc[i]) for i in sv_nums])
        if is_subtitled:
            axarr[n].set_title(['SVD ({})'.format(descr_str), 't-SNE'][n], fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # add the labels for each cat
        for cat in model.hub.probe_store.cats:
            probes_acts_2d_ids = np.where(np.asarray(probes_acts_cats) == cat)[0]
            xtext, ytext = np.median(probes_acts_2d[probes_acts_2d_ids, :], axis=0)
            txt = axarr[n].text(xtext, ytext, str(cat), fontsize=FigsConfigs.TICKLABEL_FONT_SIZE,
                                color=palette[model.hub.probe_store.cats.index(cat)])
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=FigsConfigs.LINEWIDTH, foreground="w"), PathEffects.Normal()])
        # add the labels for each probe
        if label_probe:
            for probe in model.hub.probe_store.types:
                probes_acts_2d_ids = np.where(np.asarray(model.hub.probe_store.types) == probe)[0]
                xtext, ytext = np.median(probes_acts_2d[probes_acts_2d_ids, :], axis=0)
                txt = axarr[n].text(xtext, ytext, str(probe), fontsize=FigsConfigs.TICKLABEL_FONT_SIZE)
                txt.set_path_effects([
                    PathEffects.Stroke(linewidth=FigsConfigs.LINEWIDTH, foreground="w"), PathEffects.Normal()])
    fig.tight_layout()
    return fig



embedder_name2plot_data = {embedder_name: [] for embedder_name in EMBEDDER_NAMES}
job_name2plot_data = {}
for param2val in gen_param2vals_for_completed_jobs(local=LOCAL):
    embedder_name = to_embedder_name(param2val)
    job_name = param2val['job_name']
    print('\n==================================\nUsing param2val for {}'.format(embedder_name))

    # TODO load process2_embed_mats.npy
