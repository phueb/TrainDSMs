import string
import numpy as np
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
import pandas as pd

from src import config
from src.embedders.base import EmbedderBase

CORPUS_NAME = 'childes-20180319'
VERBOSE = False
LEMMATIZE = True


def strip_pos(col):
    return col.split('-')[0]


def rename_relation(col):
    if col == 'mero':
        return 'has'
    elif col == 'attri':
        return 'is'
    else:
        return col


if __name__ == '__main__':
    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
    for vocab_size in config.Eval.vocab_sizes:
        # process BLESS data
        bless_df = pd.read_csv(config.Dirs.tasks / 'BLESS.txt', sep="\t", header=None)
        bless_df.columns = ['Concept', 'class', 'relation', 'relatum']
        bless_df['concept'] = bless_df['Concept'].apply(strip_pos)
        bless_df['relatum'] = bless_df['relatum'].apply(strip_pos)
        bless_df['relation'] = bless_df['relation'].apply(rename_relation)
        # make probes
        vocab = EmbedderBase.load_corpus_data(num_vocab=vocab_size)[1]
        assert len(vocab) == vocab_size
        probes = []
        for w in vocab:
            if len(w) > 1:
                if w[0] not in list(string.punctuation) \
                        and w[1] not in list(string.punctuation):
                    if LEMMATIZE:
                        for pos in ['noun', 'verb', 'adj']:
                            w = lemmatizer(w, pos)[0]
                            if w in concepts:
                                probes.append(w)
                    else:
                        if w in concepts:
                            probes.append(w)
        if LEMMATIZE:
            probes = set([p for p in probes if p in vocab])  # lemmas may not be in vocab
        # write to file
        for relation in ['has', 'is']:
            out_path = config.Dirs.tasks / 'features' / relation / '{}_{}.txt'.format(CORPUS_NAME, vocab_size)
            if not out_path.parent.exists():
                out_path.parent.mkdir(parents=True)
            with out_path.open('w') as f:
                print('Writing {}'.format(out_path))
                for probe in probes:
                    # get features for probe
                    bless_features = np.unique(
                        bless_df.loc[(bless_df['concept'] == probe) & (bless_df['relation'] == relation)]
                        ['relatum'].apply(to_object)).tolist()
                    mcrae_features = np.unique(
                        mcrae_df.loc[(mcrae_df['concept'] == probe) & (mcrae_df['relation'] == relation)]
                        ['relatum'].apply(to_object)).tolist()
                    # check
                    if VERBOSE:
                        for mcrae_f in mcrae_features:
                            if mcrae_f not in bless_features:
                                print('{}-{} in McRae data but not in BLESS data.'.format(probe, mcrae_f))
                    # write
                    features = ' '.join([f for f in mcrae_features + bless_features
                                         if f != probe and f in vocab])
                    if not features:
                        continue
                    line = '{} {}\n'.format(probe, features)
                    print(line.strip('\n'))
                    f.write(line)

