from pathlib import Path
import numpy as np

from src.embedders.ww_matrix import WWEmbedder
from src.embedders.wd_matrix import WDEmbedder
from src import config
from src.utils import matrix_to_simmat, print_matrix
from src.utils import w2e_to_matrix


#config.Corpora.num_vocab = None
config.WW.window_size = 1
config.WW.window_weight = 'flat'

p = Path('task_data/test_probes2.txt')
probe_list = np.loadtxt(p, dtype='str').tolist()

wwmatrix = WWEmbedder('childes-20180319')
w2e = wwmatrix.train()
mat = w2e_to_matrix(w2e)
norm_matrix, ppmi_cols = wwmatrix.norm_ppmi(mat)
svd_matrix, svd_cols = wwmatrix.reduce_svd(norm_matrix, 20)
sim_matrix = matrix_to_simmat(svd_matrix, 'cosine')
print_matrix(wwmatrix.vocab, sim_matrix, 3, probe_list, probe_list)

wdmatrix = WDEmbedder('childes-20180319')
w2e = wdmatrix.train()
mat = w2e_to_matrix(w2e)
norm_matrix, norm_cols = wdmatrix.norm_logentropy(mat)
svd_matrix, svd_cols = wdmatrix.reduce_svd(norm_matrix, 20)
sim_matrix = matrix_to_simmat(svd_matrix, 'cosine')
print_matrix(wdmatrix.vocab, sim_matrix, 3, probe_list, probe_list)
