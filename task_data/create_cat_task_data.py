import numpy as np

from src import config
from src.utils import load_corpus_data


CORPUS_NAME = 'childes-20180319'


if __name__ == '__main__':
    for vocab_size in config.Tasks.vocab_sizes:
        vocab = load_corpus_data(num_vocab=vocab_size)[1]
        for task_name in ['semantic_categorization', 'syntactic_categorization']:
            # load all probes
            in_path = config.Global.task_dir / task_name / '4.txt'
            probes, cats = np.loadtxt(in_path, dtype='str').T
            # write probes if in vocab
            out_fname = '{}_{}.txt'.format(CORPUS_NAME, vocab_size)
            out_path = config.Global.task_dir / task_name / out_fname
            print('Writing {}'.format(out_path))
            with out_path.open('w') as f:
                for probe, cat in zip(probes, cats):
                    if probe in vocab:
                        f.write('{} {}\n'.format(probe, cat))