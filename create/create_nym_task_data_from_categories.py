import numpy as np
import pandas as pd


from two_process_nlp import config


VERBOSE = True

SHUFFLE = False  # only for debugging
REMOVE_DUPLICATES_IN_WORDS_COL = False  # only for debugging
CORPUS_NAME = 'childes-20180319'
GRAM_CAT = 'adj'


def to_word_list(x):
    return pd.Series({'word_list': x['word'].tolist()})


if __name__ == '__main__':
    for vocab_size in config.Corpus.vocab_sizes:
        # load nyms data
        df = pd.read_excel('{}_nyms.xlsx'.format(CORPUS_NAME))
        if SHUFFLE:
            df['word'] = np.random.permutation(df['word'].tolist())
        if REMOVE_DUPLICATES_IN_WORDS_COL:
            df = df.drop_duplicates(subset='word', keep="last")
        df.drop_duplicates(inplace=True)
        df_filtered = df[df['gram'] == GRAM_CAT]
        grouped = df_filtered.groupby(['category', 'group'])
        df_combined = grouped.apply(to_word_list)
        df_combined.reset_index(level=1, inplace=True)  # make index 'group' a column
        print(df_combined)
        # vocab
        p = config.LocalDirs.root / '{}_{}_vocab.txt'.format(config.Corpus.name, vocab_size)
        if not p.exists():
            raise RuntimeError('{} does not exist'.format(p))
        vocab = np.loadtxt(p, 'str').tolist()
        # probes
        assert len(vocab) == vocab_size
        probes = []
        for word in df_filtered['word']:
            if word in vocab:
                probes.append(word)
        print('Num words in vocab={}\n'.format(len(probes)))
        # get both syns and ants for probes
        probe2syns = {p: [] for p in probes}
        probe2ants = {p: [] for p in probes}
        probe2cat = {}
        for cat, group in df_combined.groupby('category'):
            if VERBOSE:
                print(cat)
            word_lists = group['word_list'].values
            if not len(word_lists) == 2:
                print('WARNING: Skipping category "{}"'.format(cat))
                continue
            a = word_lists[0]
            b = word_lists[1]
            a = [p for p in a if p in vocab]
            b = [p for p in b if p in vocab]
            if VERBOSE:
                print(a)
                print(b)
            # check
            for probe in a:
                if a in b:
                    raise RuntimeError('Found "{}" on opposite loadings of category.')
            # get nyms for each probe
            excluded_syns = []
            for probe in a:
                excluded_syns.append(probe)
                probe2syns[probe] = [p for p in a if p != probe]
                probe2ants[probe] = [p for p in b]
                probe2cat[probe] = cat.upper() + '-'
            for probe in b:
                probe2syns[probe] = [p for p in b if p != probe]
                probe2ants[probe] = [p for p in a]
                probe2cat[probe] = cat.upper() + '+'
            # check
            if VERBOSE:
                print([(p, probe2syns[p]) for p in a + b])  # duplicates are allowed when duplicates are mirrored versions of existing pairs
                print([(p, probe2ants[p]) for p in a + b])  # there will always be more antonym pairs than synonym pairs
                print()
        # write to file
        for nym_type, probe2nyms in [('syn', probe2syns),
                                     ('ant', probe2ants)]:
            suffix = 'jw'
            if SHUFFLE:
                suffix += 'shuffled'
            if REMOVE_DUPLICATES_IN_WORDS_COL:
                suffix += 'unique'
            out_path = config.LocalDirs.tasks / 'nyms' / nym_type / '{}_{}_{}.txt'.format(
                CORPUS_NAME, vocab_size, suffix)
            if not out_path.parent.exists():
                out_path.parent.mkdir()
            with out_path.open('w') as f:
                print('Writing {}'.format(out_path))
                for probe, nyms in probe2nyms.items():
                    nyms = ' '.join([nym for nym in nyms
                                     if nym != probe and nym in vocab])
                    if not nyms:
                        continue
                    cat = probe2cat[probe]
                    line = '{} {} {}\n'.format(probe, nyms, cat)
                    print(line.strip('\n'))
                    f.write(line)