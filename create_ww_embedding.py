from src.embedders.ww_matrix import WWEmbedder
from src import config

config.Corpora.num_vocab = None
config.WW.window_size = 2
config.WW.window_weight = 'flat'

my_wwmatrix = WWEmbedder('test_corpus2')
w2e, size = my_wwmatrix.train()
matrix = my_wwmatrix.w2e_to_matrix(w2e)
# rowprob_matrix, rowprob_cols = my_wwmatrix.norm_rowsum(w2e)
# colprob_matrix, colprob_cols = my_wwmatrix.norm_colsum(w2e)
# tdidf_matrix, tdidf_cols = my_wwmatrix.norm_tdidf(w2e)

# print("\nEmbedding Dict")
# for word in w2e:
# 	print('{:15} {}'.format(word, w2e[word]))

print("\nCooc Matrix")
for i in range(len(matrix[:,0])):
	print('{:<15}'.format(my_wwmatrix.vocab[i]), end='')
	for j in range(len(matrix[i,:])):
		print('{:3.0f}  '.format(matrix[i,j]), end='')
	print()

# print("\nRow Prob Matrix")
# for i in range(len(rowprob_matrix[:,0])):
# 	print('{:<15}'.format(my_wwmatrix.vocab[i]), end='')
# 	for j in range(len(rowprob_matrix[i,:])):
# 		print('{:0.3f}  '.format(rowprob_matrix[i,j]), end='')
# 	print()

# print("\nCol Prob Matrix")
# for i in range(len(colprob_matrix[:,0])):
# 	print('{:<15}'.format(my_wwmatrix.vocab[i]), end='')
# 	for j in range(len(colprob_matrix[i,:])):
# 		print('{:0.3f}  '.format(colprob_matrix[i,j]), end='')
# 	print()

# print("\nTd-idf Matrix")
# for i in range(len(colprob_matrix[:,0])):
# 	print('{:<15}'.format(my_wwmatrix.vocab[i]), end='')
# 	for j in range(len(colprob_matrix[i,:])):
# 		print('{:0.3f}  '.format(colprob_matrix[i,j]), end='')
# 	print()