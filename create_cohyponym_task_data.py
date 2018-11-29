import numpy as np

from src import config
from src.embedders.base import EmbedderBase

CORPUS_NAME = 'childes-20180319'


if __name__ == '__main__':
    for vocab_size in config.Task.vocab_sizes:
        vocab = EmbedderBase.load_corpus_data(num_vocab=vocab_size)[1]
        for data_name1 in ['semantic', 'syntactic']:
            # load all probes
            in_path = config.Dirs.categories / data_name1 / 'complete.txt'
            probes, probe_cats = np.loadtxt(in_path, dtype='str').T
            cat2probes = {cat: probes[probe_cats == cat].tolist() for cat in probe_cats}
            # write probes if in vocab
            out_fname = '{}_{}.txt'.format(CORPUS_NAME, vocab_size)
            out_path = config.Dirs.tasks / 'cohyponyms' / data_name1 / out_fname
            print('Writing {}'.format(out_path))
            with out_path.open('w') as f:
                for probe, cat in zip(probes, probe_cats):
                    if probe in vocab:
                        cohyponyms = ' '.join([cohyponym for cohyponym in cat2probes[cat]
                                              if cohyponym != probe and cohyponym in vocab])
                        f.write('{} {}\n'.format(probe, cohyponyms))