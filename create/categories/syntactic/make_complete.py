import pandas as pd

from src import config

CORPUS_NAME = 'childes-20180319'
DIFFICULTY = 'easy'  # TODO  ['easy' 'medium' 'hard']
MAX = 200  # without a maximum, the write buffer fails (i think) because file is incomplete

df = pd.read_csv('{}_pos_data.txt'.format(CORPUS_NAME), sep="\t", header=None)
df.columns = ['word', 'pos', 'num1', 'num2', 'num3', 'difficulty']
pos_list = df['pos'].unique()
pos2words = {pos: [] for pos in pos_list}
for pos in pos_list:
    pos_words = df[(df['pos'] == pos) & (df['difficulty'] == DIFFICULTY)]['word'].values.tolist()[:MAX]
    pos2words[pos] = pos_words

p = config.Dirs.root / 'create' / 'categories' / 'syntactic' / '{}_complete.txt'.format(CORPUS_NAME)
with open(str(p), 'w') as f:
    for pos in pos_list:
        for probe in pos2words[pos]:
            line = '{} {}\n'.format(probe, pos.upper())
            f.write(line)
            print(line)