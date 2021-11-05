import numpy as np

from traindsms.utils import init_embedder
from traindsms.utils import w2e_to_sims
from traindsms.params import to_embedder_name

from analyze.utils import gen_param2vals_for_completed_jobs


embedder_names = ['ww', 'wd', 'sg', 'cbow', 'srn', 'lstm', 'random_normal', 'random_uniform']
embedder_name2plot_data = {embedder_name: [] for embedder_name in embedder_names}
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
    #
    col_ids = np.argmin(vocab_sims_mat, axis=1)
    print(vocab_sims_mat[np.arange(len(vocab_sims_mat)), col_ids])

    # TODO are pairs with lowest (negative) cosine sim antonyms?
    for row_id, col_id in list(enumerate(col_ids))[1000:1100]:
        print(embedder.vocab[row_id], embedder.vocab[col_id])
        print(vocab_sims_mat[row_id, col_id])
        print()