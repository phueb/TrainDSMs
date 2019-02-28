import string
import numpy as np
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
import pandas as pd

from two_process_nlp import config

CORPUS_NAME = 'childes-20180319'
VERBOSE = False
LEMMATIZE = True


def to_relation(col):
    return col.split('_')[0]


def to_object(col):
    return col.split('_')[-1]


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
    for vocab_size in config.Corpus.vocab_sizes:
        # process mcrae data
        mcrae_df = pd.read_csv(config.LocalDirs.create / 'mcrae_features.csv', index_col=False)
        mcrae_df.rename(inplace=True, columns={'Feature': 'relatum'})
        mcrae_df['concept'] = [w.split('_')[0] for w in mcrae_df['Concept']]
        print('Number of unique concept words={}'.format(len(mcrae_df['concept'].unique())))
        mcrae_df['relation'] = mcrae_df['relatum'].apply(to_relation)
        num_relations = mcrae_df['relation'].groupby(mcrae_df['relation']).count().sort_values()
        num_relations = num_relations.to_frame('frequency')
        print(num_relations)
        # process BLESS data
        bless_df = pd.read_csv(config.LocalDirs.create / 'BLESS.txt', sep="\t", header=None)
        bless_df.columns = ['concept', 'class', 'relation', 'relatum']
        bless_df['concept'] = bless_df['concept'].apply(strip_pos)
        bless_df['relatum'] = bless_df['relatum'].apply(strip_pos)
        bless_df['relation'] = bless_df['relation'].apply(rename_relation)
        # vocab
        p = config.RemoteDirs.root / '{}_{}_vocab.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
        if not p.exists():
            raise RuntimeError('{} does not exist'.format(p))
        vocab = np.loadtxt(p, 'str').tolist()
        # make probes
        concepts = mcrae_df['concept'].values.tolist() + bless_df['concept'].values.tolist()
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
            probes = set([probe for probe in probes if probe in vocab])  # lemmas may not be in vocab
        # write to file
        for relation in ['has', 'is']:
            out_path = config.LocalDirs.tasks / 'features' / relation / '{}_{}.txt'.format(CORPUS_NAME, vocab_size)
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
                    all_unique_features = set(mcrae_features + bless_features)
                    features = ' '.join([f for f in all_unique_features if f != probe and f in vocab])
                    if not features:
                        continue
                    line = '{} {}\n'.format(probe, features)
                    print(line.strip('\n'))
                    f.write(line)

