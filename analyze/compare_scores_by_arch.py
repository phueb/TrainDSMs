from scipy import stats

from two_process_nlp.aggregator import Aggregator

ag = Aggregator()
df = ag.make_df(load_from_file=True, verbose=True)

# clean df
df.drop(df[df['regime'].isin(['novice', 'control'])].index, inplace=True)
df.drop(df[df['neg_pos_ratio'] == 0.0].index, inplace=True)
df.drop(df[df['embedder'] == 'random_normal'].index, inplace=True)

print(len(df[df['arch'] == 'classifier']))
print(len(df[df['arch'] == 'comparator']))


for name, group in df.groupby('task'):

    classifier_rows = group[group['arch'] == 'classifier']
    comparator_rows = group[group['arch'] == 'comparator']

    a = classifier_rows['score'].values
    b = comparator_rows['score'].values

    # makse sure that rows are paired
    assert list(classifier_rows['embedder'].values) == list(comparator_rows['embedder'].values)
    assert list(classifier_rows['task'].values) == list(comparator_rows['task'].values)
    assert list(classifier_rows['job_name'].values) == list(comparator_rows['job_name'].values)

    t, p = stats.ttest_rel(a, b)  # paired t-test

    print(name)
    print('mean={:.4f} std={:.4f} n={}'.format(
        a.mean(), a.std(), len(a)))
    print('mean={:.4f} std={:.4f} n={}'.format(
        b.mean(), b.std(), len(b)))
    print('t={:.2f} p={:.4f}'.format(t, p))
    print()