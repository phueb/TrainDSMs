from collections import Counter

from src import config

p = config.Corpora.dir / '{}.txt'.format(config.Corpora.name)
words = p.read_text().split()
w2freq = Counter(words)