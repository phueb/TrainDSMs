



class GloVe():
    def __init__(self, param2val):

        self.embed_size = param2val['embed_size']
        self.lr = param2val['lr']
        self.num_epochs = param2val['num_epochs']
        self.window_size = param2val['window_size']

    def train(self):
        return NotImplementedError('Use original implementation')
