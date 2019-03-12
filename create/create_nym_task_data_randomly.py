import numpy as np
from itertools import cycle


from two_process_nlp import config


CORPUS_NAME = 'childes-20180319'
VERBOSE = True

NUM_ROW_WORDS = 400
NUM_COL_WORDS = 200
NUM_NYMS = [1, 10]
SUFFIX = 'random3'

if __name__ == '__main__':
    for vocab_size in config.Corpus.vocab_sizes:
        # vocab
        p = config.RemoteDirs.root / '{}_{}_vocab.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
        if not p.exists():
            raise RuntimeError('{} does not exist'.format(p))
        vocab = np.loadtxt(p, 'str').tolist()
        np.random.shuffle(vocab)
        # probes
        row_words = np.random.choice(vocab, size=NUM_ROW_WORDS, replace=False)
        # generate syns and ants for probes
        probe2syns = {}
        probe2ants = {}
        candidates = cycle([v for v in vocab if v not in row_words][:NUM_COL_WORDS])
        for rw in row_words:
            probe2syns[rw] = [next(candidates) for _ in range(
                np.random.randint(NUM_NYMS[0], NUM_NYMS[-1] + 1, size=1).item())]
            probe2ants[rw] = [next(candidates) for _ in range(
                np.random.randint(NUM_NYMS[0], NUM_NYMS[-1] + 1, size=1).item())]

        # write to file
        for nym_type, probe2nyms in [('syn', probe2syns),
                                     ('ant', probe2ants)]:
            out_path = config.LocalDirs.tasks / 'nyms' / nym_type / '{}_{}_{}.txt'.format(
                CORPUS_NAME, vocab_size, SUFFIX)
            if not out_path.parent.exists():
                out_path.parent.mkdir()
            with out_path.open('w') as f:
                print('Writing {}'.format(out_path))
                for probe, nyms in probe2nyms.items():
                    nyms = ' '.join([nym for nym in nyms
                                     if nym != probe and nym in vocab])
                    if not nyms:
                        continue
                    line = '{} {}\n'.format(probe, nyms)
                    print(line.strip('\n'))
                    f.write(line)