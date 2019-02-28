import numpy as np

from two_process_nlp import config

CORPUS_NAME = 'childes-20180319'


if __name__ == '__main__':
    for vocab_size in config.Corpus.vocab_sizes:

        # TODO load xlsx file and parse it




        # vocab
        p = config.RemoteDirs.root / '{}_{}_vocab.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
        if not p.exists():
            raise RuntimeError('{} does not exist'.format(p))
        vocab = np.loadtxt(p, 'str').tolist()
        # probes
        assert len(vocab) == vocab_size

        # TODO exclude words if not in vocab

        #
        for nym_type in ['syn', 'ant']:
            out_path = config.LocalDirs.tasks / 'nyms' / nym_type / '{}_{}_jw.txt'.format(CORPUS_NAME, vocab_size)
            if not out_path.parent.exists():
                out_path.parent.mkdir()
            with out_path.open('w') as f:

                # TODO nyms

                # write to file
                print('Writing {}'.format(out_path))
                for probe, nyms in zip(probes, nyms):
                    nyms = ' '.join([nym for nym in nyms
                                     if nym != probe and nym in vocab])
                    if not nyms:
                        continue
                    line = '{} {}\n'.format(probe, nyms)
                    print(line.strip('\n'))
                    f.write(line)