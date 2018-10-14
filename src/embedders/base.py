import numpy as np
import yaml
import datetime

from src import config
from src.utils import load_corpus_data
from src.utils import matrix_to_w2e


numeric_docs, vocab, w2freq = load_corpus_data()


class EmbedderBase(object):
    def __init__(self):
        pass  # TODO remove this?

    @staticmethod
    def save_params(params):
        fname = None  # TODO implement
        p = config.Dirs.params / fname
        with p.open('w', encoding='utf8') as outfile:
            yaml.dump(params, outfile, default_flow_style=False, allow_unicode=True)


    @property
    def embeddings_fname(self):
        time_of_init = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
        return '{}.txt'.format(time_of_init)  # TODO need model type information in name?

    @property
    def numeric_docs(self):
        return numeric_docs

    @property
    def vocab(self):
        return vocab

    @property
    def w2freq(self):
        return w2freq

    def has_embeddings(self):
        p = config.Dirs.embeddings / self.embeddings_fname
        return True if p.exists() else False

    @staticmethod
    def check_consistency(mat):
        # size check
        assert mat.shape[1] > 1
        print('Inf Norm of embeddings = {:.1f}'.format(np.linalg.norm(mat, np.inf)))

    def load_w2e(self):
        mat = np.loadtxt(config.Dirs.embeddings / self.embeddings_fname, dtype='str', comments=None)
        vocab = mat[:, 0]
        embed_mat = mat[:, 1:].astype('float')
        w2e = matrix_to_w2e(embed_mat, vocab)
        self.check_consistency(embed_mat)
        return w2e

    def save_w2e(self, w2e):  # TODO serializing is faster (pickle, numpy)
        p = config.Dirs.embeddings / self.embeddings_fname
        with p.open('w') as f:
            for probe, embedding in sorted(w2e.items()):
                embedding_str = ' '.join(np.around(embedding, config.Embeddings.precision).astype(np.str).tolist())
                f.write('{} {}\n'.format(probe, embedding_str))