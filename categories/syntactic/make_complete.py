import pandas as pd

from src import config

DIFFICULTY = 'easy'  # TODO

df = pd.read_csv('spacy_pos_data_childes.txt', sep="\t",
                 header=None)
df.columns = ['word', 'pos', 'num1', 'num2', 'num3', 'difficulty']

pos_list = df['pos'].unique()
pos2words = {pos: [] for pos in pos_list}
for pos in pos_list:
    pos_words = df[(df['pos'] == pos) & (df['difficulty'] == DIFFICULTY)]['word'].values.tolist()
    pos2words[pos] = pos_words
print(pos2words)

p = config.Dirs.tasks / 'cohyponyms' / 'syntactic' / 'complete.txt'
with p.open('w') as f:
    for pos, pos_words in pos2words.items():
        for probe in pos_words:
            relata = [w for w in pos2words[pos] if w != probe]
            relata = ' '.join(relata)
            line = '{} {}\n'.format(probe, relata)
            print(line.strip('\n'))
            f.write(line)