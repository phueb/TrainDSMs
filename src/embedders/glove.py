from glove import Glove
from glove import Corpus

from src import config
from src.embedders.base import EmbedderBase
from src.params import GloveParams


class GloveEmbedder(EmbedderBase):
    def __init__(self, param2ids, param2val):  # TODO put glove into count class?
        super().__init__(param2val)
        self.embed_size = GloveParams.embed_size[param2ids.embed_size]
        self.lr = GloveParams.lr[param2ids.lr]
        self.num_epochs = GloveParams.num_epochs[param2ids.num_epochs]
        self.window_size = GloveParams.window_size[param2ids.window_size]
        #
        self.name = 'glove'

    def train(self):
        # get co-occurences
        corp = Corpus()
        corp.fit(self.docs, window=self.window_size)
        print('Dict size: %s' % len(corp.dictionary))
        print('Collocations: %s' % corp.matrix.nnz)
        # train
        print('Training the GloVe model')
        glove = Glove(no_components=self.embed_size, learning_rate=self.lr)
        glove.fit(corp.matrix, epochs=self.num_epochs,
                  no_threads=config.Glove.num_threads, verbose=True)
        glove.add_dictionary(corp.dictionary)
        #
        self.w2e = {w: glove.word_vectors[i] for w, i in corp.dictionary.items()}
