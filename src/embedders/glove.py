
from src import config
from src.embedders.base import EmbedderBase
from src.params import GloveParams


class GloveEmbedder(EmbedderBase):
    def __init__(self, param2ids, param2val):
        super().__init__(param2val)
        self.glove_type = GloveParams.glove_type[param2ids.glove_type]
        self.embed_size = GloveParams.embed_size[param2ids.embed_size]
        self.lr = GloveParams.lr[param2ids.lr]
        self.num_epochs = GloveParams.num_epochs[param2ids.num_epochs]
        self.window_size = GloveParams.window_size[param2ids.window_size]
        #
        self.name = self.glove_type

    def train(self):
        return NotImplementedError('Use original implementation')
