from src.embedders.ww_matrix import WWEmbedder
from src.embedders.wd_matrix import WDEmbedder
from src import config

#config.Corpora.num_vocab = None
config.WW.window_size = 1
config.WW.window_weight = 'flat'

f = open('task_data/test_probes2.txt')
probe_list = []
for line in f:
	probe_list.append(line.strip().strip('\n').strip())
f.close()

wwmatrix = WWEmbedder('childes-20180319')
w2e, size = wwmatrix.train()
norm_matrix, ppmi_cols = wwmatrix.norm_ppmi(w2e)
svd_matrix, svd_cols = wwmatrix.reduce_svd(norm_matrix, 20)
sim_matrix = wwmatrix.sim_matrix(svd_matrix, 'cosine')
wwmatrix.print_matrix(sim_matrix, 3, probe_list, probe_list)

wdmatrix = WDEmbedder('childes-20180319')
w2e, size = wdmatrix.train()
norm_matrix, norm_cols = wdmatrix.norm_logentropy(w2e)
svd_matrix, svd_cols = wdmatrix.reduce_svd(norm_matrix, 20)
sim_matrix = wdmatrix.sim_matrix(svd_matrix, 'cosine')
wdmatrix.print_matrix(sim_matrix, 3, probe_list, probe_list)
