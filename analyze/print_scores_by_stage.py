

from two_process_nlp.aggregator import Aggregator

ag = Aggregator()
df = ag.make_df(load_from_file=True, verbose=True)

# clean df
df.drop(df[df['neg_pos_ratio'] == 0.0].index, inplace=True)
df.drop(df[df['embedder'] == 'random_normal'].index, inplace=True)


avg_nov_score = df[df['process'] == 'novice']['score'].mean()
std_nov_score = df[df['process'] == 'novice']['score'].std()
avg_exp_score = df[df['process'] == 'expert']['score'].mean()
std_exp_score = df[df['process'] == 'expert']['score'].std()

print(avg_nov_score)
print(std_nov_score)
print()
print(avg_exp_score)
print(std_exp_score)