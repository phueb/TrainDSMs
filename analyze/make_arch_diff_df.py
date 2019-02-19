import pickle


from two_stage_nlp.aggregator import Aggregator
from two_stage_nlp import config
from analyze.utils import to_diff_df


def load_column_from_file(which):
    p = config.Dirs.remote_root / 'job_name2{}_probe_sim_data.pkl'.format(which)
    with p.open('rb') as f:
        job_name2probe_sim_data = pickle.load(f)
    res =[job_name2probe_sim_data[row['job_name']][0][row['task']]
          for n, row in diff_df.iterrows()]
    return res


# make diff_df
ag = Aggregator()
df = ag.make_df(load_from_file=True, verbose=True)
diff_df = to_diff_df(df)

# add data about sim per embedder and per task
diff_df['all_probe_sim'] = load_column_from_file('all')
diff_df['pos_probe_sim'] = load_column_from_file('pos')

# save
p = config.Dirs.remote_root / 'diff_scores.csv'
diff_df.to_csv(p)

