from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

from two_stage_nlp.job_utils import init_embedder
from two_stage_nlp.job_utils import w2e_to_sims
from two_stage_nlp.architectures import comparator
from two_stage_nlp.evaluators.matching import Matching
from two_stage_nlp.params import to_embedder_name

from analyze.utils import gen_param2vals_for_completed_jobs
from analyze.utils import to_label


WHICH_PAIRS = 'all'  # TODO implement negative pairs
BY_EMBEDDER = False


evaluators = [
    Matching(comparator, 'cohyponyms', 'semantic'),
    Matching(comparator, 'cohyponyms', 'syntactic'),
    Matching(comparator, 'features', 'is'),
    Matching(comparator, 'features', 'has'),
    Matching(comparator, 'nyms', 'syn', suffix='_jw'),
    Matching(comparator, 'nyms', 'ant', suffix='_jw'),
    Matching(comparator, 'hypernyms'),
    Matching(comparator, 'events')
]

embedder_names = ['ww', 'wd', 'sg', 'cbow', 'srn', 'lstm', 'random_normal', 'random_uniform']
data_names = [ev.data_name for ev in evaluators]


data_name2mean_sum = {data_name: 0.0 for data_name in data_names}
embedder_name2plot_data = {embedder_name: [] for embedder_name in embedder_names}
job_name2plot_data = {}
for param2val in gen_param2vals_for_completed_jobs():
    embedder_name = to_embedder_name(param2val)
    job_name = param2val['job_name']
    print('\n==================================\nUsing param2val for {}'.format(embedder_name))
    embedder = init_embedder(param2val)
    embedder.load_w2e()
    # tasks (data_name refers to task + suffix)
    data_name2mean = {}
    data_name2std = {}
    for ev in evaluators:
        data_name = ev.data_name
        print(data_name)
        # probes
        vocab_sims_mat = np.zeros(1)  # dummy
        all_eval_probes, all_eval_candidates_mat = ev.make_all_eval_data(vocab_sims_mat, embedder.vocab)
        ev.row_words, ev.col_words, ev.eval_candidates_mat = ev.downsample(
            all_eval_probes, all_eval_candidates_mat)
        # calc sims
        if WHICH_PAIRS == 'all':
            mean_sim = w2e_to_sims(embedder.w2e, ev.row_words, ev.col_words).mean()
            std_sim = w2e_to_sims(embedder.w2e, ev.row_words, ev.col_words).std()
        elif WHICH_PAIRS == 'pos':
            pos_sims = []
            for probe, relata in ev.probe2relata.items():
                a = np.expand_dims(embedder.w2e[probe], axis=0)
                b = np.asarray([embedder.w2e[r] for r in relata])
                pos_sims_for_probe = cosine_similarity(a, b)
                pos_sims.extend(pos_sims_for_probe.flatten())
            mean_sim = np.mean(pos_sims)
            std_sim = np.std(pos_sims)
        elif WHICH_PAIRS == 'neg':
            raise NotImplementedError
        else:
            raise AttributeError('Invalid arg to "WHICH_PAIRS".')
        # collect
        data_name2mean[data_name] = mean_sim
        data_name2std[data_name] = std_sim
        #
        data_name2mean_sum[data_name]  += mean_sim
    # collect
    plot_data = (data_name2mean, data_name2std)
    embedder_name2plot_data[embedder_name].append(plot_data)
    job_name2plot_data[job_name] = plot_data

# figure
embedder_name2color = {embedder_name: plt.cm.get_cmap('tab10')(n)
                       for n, embedder_name in enumerate(embedder_names)}

fig, ax = plt.subplots(1, figsize=(10, 5), dpi=300)
plt.title('Cosine similarities between {} probe pairs in task'.format(WHICH_PAIRS))
sorted_data_names = sorted(data_names, key=data_name2mean_sum.get, reverse=True)
num_x = len(sorted_data_names)
x = np.arange(num_x)
ax.set_xticks(x)
ax.set_xticklabels([to_label(dn).replace('_', '\n') for dn in sorted_data_names])
ax.set_ylabel('Cosine Similarity')
ax.set_xlabel('Tasks')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
if BY_EMBEDDER:
    for embedder_name, plot_data in embedder_name2plot_data.items():
        for n, (data_name2mean, data_name2std) in enumerate(plot_data):
            label = embedder_name if n == 0 else None
            color = embedder_name2color[embedder_name]
            means = np.asarray([data_name2mean[dn] for dn in sorted_data_names])
            stds = np.asarray([data_name2std[dn] for dn in sorted_data_names])
            ax.plot(x, means, label=label, color=color, zorder=3, linewidth=2)
            ax.fill_between(x, means - stds / 2, means + stds / 2, facecolor=color, alpha=0.05)
else:
    for n, dn in enumerate(sorted_data_names):
        ys = np.asarray([plot_data[0][0][dn] if plot_data else np.nan
                         for plot_data in embedder_name2plot_data.values()])
        ax.bar(x=n,
               height=np.nanmean(ys),
               yerr=np.nanstd(ys),
               edgecolor='black',
               width=1.0)

ax.legend(loc='best', frameon=False)
plt.tight_layout()
plt.show()


