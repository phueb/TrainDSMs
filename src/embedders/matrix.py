from src.embedders import EmbedderBase


class MatrixEmbedder(EmbedderBase):
    def __init__(self, corpus_name, name_suffix):
        super().__init__(corpus_name, '{}'.format(name_suffix))