import pandas as pd

from two_stage_nlp.aggregator import Aggregator
from two_stage_nlp import config

ag = Aggregator()
df = ag.make_df(load_from_file=True, verbose=True)

df.drop(df[df['stage'].isin(['novice', 'control'])].index, inplace=True)
df.drop(df[df['neg_pos_ratio'] == 0.0].index, inplace=True)
del df['corpus']
del df['num_vocab']
del df['embed_size']
del df['Unnamed: 0']
del df['evaluation']
del df['param_name']
del df['stage']
del df['neg_pos_ratio']
del df['num_epochs_per_row_word']

print([name for name, _ in df.groupby('arch')])
df1, df2 = [x for _, x in df.groupby('arch')]

assert list(df1['embedder'].values) == list(df2['embedder'].values)
assert list(df1['task'].values) == list(df2['task'].values)
assert list(df1['job_name'].values) == list(df2['job_name'].values)

print(df1['score'])
print(df2['score'])


df1['diff_score'] = df1['score'].values - df2['score'].values
del df1['arch']
del df1['score']
print(df1)
print(df1.columns)

p = config.Dirs.remote_root / 'diff_scores.csv'
df1.to_csv(p)