from pathlib import Path
import numpy as np

from src.embedders.ww_matrix import WWEmbedder
from src.embedders.wd_matrix import WDEmbedder
from src import config
from src.utils import matrix_to_simmat, print_matrix
from src.utils import w2e_to_matrix

# vocab_sizes = [4096,8192,16384]
# window_sizes = [1,2,4,8,16]
# window_weights = ["flat", 'linear']
# window_types = ['forward', 'backward', 'summed', 'concatenate']
# norm_types = ['none', 'row_sum', 'col_sum', 'log-entropy', 'td-idf', 'ppmi']
# reduce_types = ['none', 'svd']
# reduce_sizes = [16, 32, 64, 128, 256, 512]

vocab_sizes = [4096]
window_sizes = [1]
window_weights = ["flat"]
window_types = ['forward', 'backward']
norm_types = ['none', 'row_logentropy', 'ppmi']
reduce_types = ['none','svd']
reduce_sizes = [32,512]

p = Path('task_data/test_probes2.txt')
probe_list = np.loadtxt(p, dtype='str').tolist()

# corpus_name = 'childes100d'
corpus_name = 'childes-20180319'

for vocab_size in vocab_sizes:
    config.Corpora.num_vocab = vocab_size

    for norm_type in norm_types:
        for reduce_type in reduce_types:

            print(norm_type, reduce_type)

            if reduce_type != 'none':
                for reduce_size in reduce_sizes:

                    wd_embedder_name = 'wd_v' + str(vocab_size) + '_n-' + norm_type + '_r-' + reduce_type + '_s' + str(reduce_size)
                    wdmatrix = WDEmbedder(corpus_name, wd_embedder_name)
                    w2e = wdmatrix.train(norm_type, reduce_type, reduce_size)
                    wdmatrix.save(w2e)

                    for window_type in window_types:
                        for window_size in window_sizes:
                            for window_weight in window_weights:
                                ww_embedder_name = 'ww_v' + str(vocab_size) + '_wt-' + window_type + \
                                                   '_ws-' + str(window_size) + '_ww-' + window_weight + \
                                                   '_n-' + norm_type + '_r-' + reduce_type + '_s' + str(reduce_size)
                                wwmatrix = WWEmbedder(corpus_name, ww_embedder_name)
                                w2e = wwmatrix.train(window_type, window_size, window_weight, norm_type, reduce_type,
                                                     reduce_size)
                                wwmatrix.save(w2e)
            else:
                reduce_size = 0
                wd_embedder_name = 'wd_v' + str(vocab_size) + '_n-' + norm_type + '_r-' + reduce_type + '_s' + str(reduce_size)
                wdmatrix = WDEmbedder(corpus_name, wd_embedder_name)
                w2e = wdmatrix.train(norm_type, reduce_type, reduce_size)
                wdmatrix.save(w2e)

                for window_type in window_types:
                    for window_size in window_sizes:
                        for window_weight in window_weights:
                            ww_embedder_name = 'ww_v' + str(vocab_size) + '_wt-' + window_type + \
                                               '_ws' + str(window_size) + '_ww-' + window_weight + \
                                               '_n-' + norm_type + '_r-' + reduce_type + '_s' + str(reduce_size)
                            wwmatrix = WWEmbedder(corpus_name, ww_embedder_name)
                            w2e = wwmatrix.train(window_type, window_size, window_weight, norm_type, reduce_type,
                                                 reduce_size)
                            wwmatrix.save(w2e)







