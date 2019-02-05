from itertools import product

from two_stage_nlp import config


CORPUS_NAME = config.Corpus.name
NUM_VOCAB = config.Corpus.num_vocab

# for p in (config.Dirs.tasks / 'cohyponyms' / 'semantic').rglob('{}_{}.txt'.format(CORPUS_NAME, NUM_VOCAB)):
# for p in (config.Dirs.tasks / 'hypernyms').rglob('{}_{}.txt'.format(CORPUS_NAME, NUM_VOCAB)):
for p in config.Dirs.tasks.rglob('{}_{}*.txt'.format(CORPUS_NAME, NUM_VOCAB)):

    with p.open('r') as f:
        lines = f.read().splitlines()  # removes '\n' newline character

    pairs = set()
    pairs.update()
    num_possible = 0
    for line in lines:
        probe = line.split()[0]
        relata = line.split()[1:]
        pairs.update(list(product([probe], relata)))
        num_possible += len(relata)
    print(p.relative_to(config.Dirs.tasks))
    print('Num pairs={:,}'.format(len(pairs)))
    print('Num possible={:,}'.format(num_possible))
    diff = num_possible - len(pairs)
    if diff > 0:
        print('WARNING: Difference={}. Duplicates exist'.format(diff))
    print()