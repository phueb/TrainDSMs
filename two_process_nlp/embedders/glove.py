
from two_process_nlp.embedders.base import EmbedderBase


class GloveEmbedder(EmbedderBase):
    def __init__(self, param2val):
        super().__init__(param2val['param_name'], param2val['job_name'])
        self.glove_type = param2val['glove_type']
        self.embed_size = param2val['embed_size']
        self.lr = param2val['lr']
        self.num_epochs = param2val['num_epochs']
        self.window_size = param2val['window_size']
        #
        self.name = self.glove_type

    def train(self):
        return NotImplementedError('Use original implementation')
