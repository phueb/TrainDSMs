import string
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
import pandas as pd

from src import config
from src.embedders.base import EmbedderBase

CORPUS_NAME = 'childes-20180319'
VERBOSE = True
LEMMATIZE = True

RELATION = 'is'  # use 'is' or 'has'


def to_relation(col):
    l = col.split('_')
    return l[0]


def to_object(col):
    l = col.split('_')
    return l[-1]


probe2features = {'is': {'jar': ['round'],
                         'candle': ['hot'],
                         'basket': ['useful'],
                         'buggy': ['fast'],
                         'vest': ['comfortable'],
                         'crayon': ['colorful'],
                         'cat': ['alive'],
                         'radio': ['loud', 'electronic'],
                         'door': ['tall', 'heavy', 'creaky'],
                         'squid': ['alive', 'wet', 'dangerous'],
                         }}


if __name__ == '__main__':
    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
    for vocab_size in config.Task.vocab_sizes:
        # process features data
        in_path = config.Dirs.tasks / 'features' / 'mcrae_features.csv'
        df = pd.read_csv(in_path, index_col=False)
        concepts = [w.split('_')[0] for w in df['Concept']]
        df['concept'] = concepts
        print('Number of unique concept words={}'.format(len(df['concept'].unique())))
        df['relation'] = df['Feature'].apply(to_relation)
        num_relations = df['relation'].groupby(df['relation']).count().sort_values()
        num_relations = num_relations.to_frame('frequency')
        print(num_relations)
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
        out_path = config.Dirs.tasks / 'features' / RELATION / '{}_{}.txt'.format(CORPUS_NAME, vocab_size)
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True)
        with out_path.open('w') as f:
            print('Writing {}'.format(out_path))
            for probe in probes:
                features = df.loc[(df['concept'] == probe) & (df['relation'] == RELATION)]['Feature'].apply(
                    to_object).tolist()
                if not features:
                    continue
                line = probe
                not_normed_features = probe2features[RELATION][probe] if probe in probe2features[RELATION] else []
                for feature in features + not_normed_features:
                    if feature not in vocab or feature == probe:
                        continue
                    line += ' {}'.format(feature)
                if ' ' in line:  # check that features were added to line
                    f.write(line + '\n')
                    if VERBOSE:
                        print(line)
                else:
                    print('\t"{}" has no features'.format(probe))