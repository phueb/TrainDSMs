import numpy as np
import matplotlib.pyplot as plt

from two_process_nlp.job_utils import init_embedder
from two_process_nlp.job_utils import w2e_to_sims
from two_process_nlp.params import to_embedder_name

from analyze.utils import gen_param2vals_for_completed_jobs


EMBEDDER_NAMES = ['ww', 'wd', 'sg', 'cbow', 'srn', 'lstm', 'random_normal', 'random_uniform']


embedder_name2plot_data = {embedder_name: [] for embedder_name in EMBEDDER_NAMES}
job_name2plot_data = {}
for param2val in gen_param2vals_for_completed_jobs():
    embedder_name = to_embedder_name(param2val)
    job_name = param2val['job_name']
    print('\n==================================\nUsing param2val for {}'.format(embedder_name))
    embedder = init_embedder(param2val)
    embedder.load_w2e()
    #
    vocab_sims_mat = w2e_to_sims(embedder.w2e, embedder.vocab, embedder.vocab)
    embedder_name2plot_data[embedder_name].append((vocab_sims_mat.mean(), vocab_sims_mat.std()))
    print(embedder_name)
    print(vocab_sims_mat.mean())
    print(vocab_sims_mat.std())

# figure
embedder_name2color = {embedder_name: plt.cm.get_cmap('tab10')(n)
                       for n, embedder_name in enumerate(EMBEDDER_NAMES)}
fig, ax = plt.subplots(1, figsize=(10, 5), dpi=300)
plt.title('Cosine similarities between all pairs in vocab')
num_x = len(EMBEDDER_NAMES)
x = np.arange(num_x)
ax.set_xticks(x)
sorted_embedder_names = list(zip(*sorted(embedder_name2plot_data.items(), key=lambda i: i[1])))[0]
ax.set_xticklabels(sorted_embedder_names)
ax.set_ylabel('Cosine Similarity')
ax.set_xlabel('Embedder')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
for n, embedder_name in enumerate(sorted_embedder_names):
    color = embedder_name2color[embedder_name]
    plot_data = embedder_name2plot_data[embedder_name]
    ys = [pd[0] for pd in plot_data]  # only mean
    print(embedder_name)
    print(ys)
    ax.bar(x=n,
           height=np.nanmean(ys),
           yerr=np.nanstd(ys),
           edgecolor='black',
           width=1.0)
plt.tight_layout()
plt.show()


