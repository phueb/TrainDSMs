import string
import numpy as np
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
import pandas as pd

from two_process_nlp import config
from two_process_nlp.embedders.base import EmbedderBase

CORPUS_NAME = 'childes-20180319'
LEMMATIZE = True


def strip_pos(col):
    return col.split('-')[0]


if __name__ == '__main__':
    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
    for vocab_size in config.Corpus.vocab_sizes:
        # process BLESS data
        bless_df = pd.read_csv(config.LocalDirs.tasks / 'BLESS.txt', sep="\t", header=None)
        bless_df.columns = ['concept', 'class', 'relation', 'relatum']
        bless_df['concept'] = bless_df['concept'].apply(strip_pos)
        bless_df['relatum'] = bless_df['relatum'].apply(strip_pos)
        # make probes
        concepts = bless_df['concept'].values.tolist()
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
        out_path = config.LocalDirs.tasks / 'events' / '{}_{}.txt'.format(CORPUS_NAME, vocab_size)
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True)
        with out_path.open('w') as f:
            print('Writing {}'.format(out_path))
            for probe in probes:
                # get relata for probe
                bless_relata = np.unique(
                    bless_df.loc[(bless_df['concept'] == probe) & (bless_df['relation'] == 'event')]
                    ['relatum']).tolist()
                # write
                relata = ' '.join([f for f in bless_relata
                                   if f != probe and f in vocab])
                if not relata:
                    continue
                line = '{} {}\n'.format(probe, relata)
                print(line.strip('\n'))
                f.write(line)

