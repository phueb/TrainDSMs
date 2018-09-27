from src.embedders import EmbedderBase


class MatrixEmbedder(EmbedderBase):
    def __init__(self, corpus_name, name_suffix):
        super().__init__(corpus_name, '{}'.format(name_suffix))


# TODO implement below - what o use as sliding window?

import numpy as np
from collections import deque

VOCAB_SIZE = 4096
LINES = '''
The child plays with toy1 . The child plays with toy2 . The child plays with toy3 .
The man eats food1 . The man eats food2 . The man eats food3 .
The woman read book1 . The woman eats book2 . The woman eats book3 .'''


matrix = np.zeros([VOCAB_SIZE, VOCAB_SIZE], float)


window = []
linecounter = 1
for line in LINES:

    token_list = (line.strip().strip('\n').strip()).split()

    for token in token_list:
        if token in self.vocab_index_dict:
            window.append(token)
        else:
            window.append("UNKNOWN")

        if len(window) >= self.window_size + 1:

            if window[0] in self.vocab_index_dict:
                word1_index = self.vocab_index_dict[window[0]]

                for target_range in range(len(window) - 1):

                    if window[target_range + 1] in self.vocab_index_dict:
                        word2_index = self.vocab_index_dict[window[target_range + 1]]

                        if window_weight == "linear":
                            matrix[word1_index, word2_index] += window_size - target_range
                        elif window_weight == "flat":
                            matrix[word1_index, word2_index] += 1
        window = window[1:]

    while len(window) > 0:

        if window[0] in self.vocab_index_dict:
            word1_index = self.vocab_index_dict[window[0]]

            for target_range in range(len(window) - 1):

                if window[target_range + 1] in self.vocab_index_dict:
                    word2_index = self.vocab_index_dict[window[target_range + 1]]

                    if window_weight == "linear":
                        matrix[word1_index, word2_index] += window_size - target_range
                    elif window_weight == "flat":
                        matrix[word1_index, word2_index] += 1
        window = window[1:]

    if linecounter % 100 == 0:
        print("        Counted co-occurrences for document %s/%s" % (linecounter, self.num_docs))
    linecounter += 1