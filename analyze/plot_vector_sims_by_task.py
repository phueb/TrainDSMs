from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

from two_stage_nlp.job_utils import init_embedder
from two_stage_nlp.job_utils import w2e_to_sims
from two_stage_nlp.architectures import comparator
from two_stage_nlp.evaluators.matching import Matching
from two_stage_nlp.params import to_embedder_name

from analyze.utils import gen_param2vals_for_completed_jobs


WHICH_PAIRS = 'pos'


def full_name_to_task_name(full_name):
    return '\n'.join(full_name.split('_')[2:])


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
task_names = [full_name_to_task_name(ev.full_name) for ev in evaluators]


task_name2mean_sum = {task_name: 0.0 for task_name in task_names}
embedder_name2plot_data = {embedder_name: [] for embedder_name in embedder_names}
for param2val in gen_param2vals_for_completed_jobs():
    embedder_name = to_embedder_name(param2val)
    print('\n==================================\nUsing param2val for {}'.format(embedder_name))
    embedder = init_embedder(param2val)
    embedder.load_w2e()
    # tasks
    task2mean = {}
    task2std = {}
    for ev in evaluators:
        task_name = full_name_to_task_name(ev.full_name)
        print(task_name)
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
        task2mean[task_name] = mean_sim
        task2std[task_name] = std_sim
        #
        task_name2mean_sum[task_name]  += mean_sim
    # collect
    plot_data = (task2mean, task2std)
    embedder_name2plot_data[embedder_name].append(plot_data)


# figure
embedder_name2color = {embedder_name: plt.cm.get_cmap('tab10')(n)
                       for n, embedder_name in enumerate(embedder_names)}
fig, ax = plt.subplots(1, figsize=(20, 10), dpi=200)
plt.title('Cosine similarities between {} probe pairs by task'.format(WHICH_PAIRS))
sorted_task_names = sorted(task_names, key=task_name2mean_sum.get)
num_x = len(sorted_task_names)
x = np.arange(num_x)
ax.set_xticks(x)
ax.set_xticklabels(sorted_task_names)
ax.set_ylabel('Cosine Similarity')
ax.set_xlabel('Tasks')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
for embedder_name, plot_data in embedder_name2plot_data.items():
    for task2mean, task2std in plot_data:
        color = embedder_name2color[embedder_name]
        means = np.asarray([task2mean[task_name] for task_name in sorted_task_names])
        stds = np.asarray([task2std[task_name] for task_name in sorted_task_names])
        ax.plot(x, means, label=embedder_name, color=color, zorder=3, linewidth=2)
        ax.fill_between(x, means - stds / 2, means + stds / 2, facecolor=color, alpha=0.1)
ax.legend(loc='best')
plt.tight_layout()
plt.show()


# TODO show plot for neg, pos, and all pairs