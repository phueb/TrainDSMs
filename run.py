import numpy as np

from src import config
from src.expert_models.cat_classifier import CatClassifier
from src.embedding_models.lstm import LSTMEmbedder

for e in [LSTMEmbedder(config.Corpus.name),
          LSTMEmbedder(config.Corpus.name)]:

    # stage 1
    if e.has_embeddings():
        embedding_mat = e.train_vectors()
    else:
        embedding_mat = e.load_vecs()

    # stage 2
    sem_cat_classifier = CatClassifier('semantic', embedding_mat)
    syn_cat_classifier = CatClassifier('semantic', embedding_mat)




