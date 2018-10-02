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

my_wwmatrix = WWEmbedder('childes-20180319')

w2e, size = my_wwmatrix.train()
matrix = my_wwmatrix.w2e_to_matrix(w2e)
rowprob_matrix, rowprob_cols = my_wwmatrix.norm_rowsum(w2e)
colprob_matrix, colprob_cols = my_wwmatrix.norm_colsum(w2e)
tdidf_matrix, tdidf_cols = my_wwmatrix.norm_tdidf(w2e)
ppmi_matrix, ppmi_cols = my_wwmatrix.norm_ppmi(w2e)
logentropy_matrix, logentropy_cols = my_wwmatrix.norm_logentropy(w2e)
#
svd_matrix, svd_cols = my_wwmatrix.reduce_svd(ppmi_matrix, 20)
# rva_matrix, rva_cols = my_wwmatrix.reduce_rva(ppmi_matrix, 2, 0, 1)
#
sim_matrix = my_wwmatrix.sim_matrix(svd_matrix, 'cosine')
#
#my_wwmatrix.print_matrix(matrix, 0, probe_list)
# my_wwmatrix.print_matrix(ppmi_matrix, 3, probe_list)
my_wwmatrix.print_matrix(svd_matrix, 3, probe_list)
my_wwmatrix.print_matrix(sim_matrix, 3, probe_list, probe_list)