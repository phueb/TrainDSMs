from src.embedders import EmbedderBase


class LSTMEmbedder(EmbedderBase):
    def __init__(self, corpus_name, ):
        super().__init__(corpus_name, 'lstm')