import numpy as np

from src import config


class EmbedderBase(object):
    def __init__(self, corpus_name, name):
        self.corpus_name = corpus_name
        self.name = name

    @property
    def embeddings_fname(self):
        return '{}_{}.txt'.format(self.corpus_name, self.name)

    @property
    def embeddings_dir(self):
        return config.Embeddings.dir

    def has_embeddings(self):
        p = self.embeddings_dir / self.embeddings_fname
        return True if p.exists() else False

    def check_consistency(self, mat):  # TODO different models may produce embeddings that benefit from different expert hyperparameters
        # size check
        assert mat.shape[1] == config.Global.embed_size
        # norm check
        assert np.max(mat) <= 1.0
        assert np.min(mat) >= -1.0
        print('Inf Norm of embeddings = {:.1f}'.format(np.linalg.norm(mat, np.inf)))

    def load_w2e(self):
        mat = np.loadtxt(self.embeddings_dir / self.embeddings_fname, dtype='str', comments=None)
        embed_mat = mat[:, 1:].astype('float')
        w2e = {probe: embedding for probe, embedding in zip(mat[:, 0], embed_mat)}
        self.check_consistency(embed_mat)
        return w2e

    def save(self, w2e):  # TODO serializing is faster (pickle, numpy)
        p = self.embeddings_dir / self.embeddings_fname
        with p.open('w') as f:
            for probe, embedding in sorted(w2e.items()):
                embedding_str = ' '.join(embedding.astype(np.str).tolist())
                f.write('{} {}\n'.format(probe, embedding_str))