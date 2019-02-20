
from two_stage_nlp import config
from two_stage_nlp.aggregator import Aggregator


def gen_param2vals_for_completed_jobs():
    for location in config.Dirs.remote_runs.glob('**/*num*'):
        param_name, job_name = location.parts[-2:]
        param2val = Aggregator.load_param2val(param_name)
        param2val['job_name'] = job_name
        yield param2val


def to_label(s):
    if s == 'nyms_syn_jw' or s == s == 'nyms_syn':
        return 'synonyms'
    elif s == 'nyms_ant_jw' or s == s == 'nyms_ant':
        return 'antonyms'
    else:
        return s


def to_diff_df(df):
    df.drop(df[df['stage'].isin(['novice', 'control'])].index, inplace=True)
    df.drop(df[df['neg_pos_ratio'] == 0.0].index, inplace=True)
    df.drop(df[df['embedder'] == 'random_normal'].index, inplace=True)
    del df['corpus']
    del df['num_vocab']
    del df['embed_size']
    del df['evaluation']
    del df['param_name']
    del df['stage']
    del df['neg_pos_ratio']
    del df['num_epochs_per_row_word']
    #
    df1, df2 = [x for _, x in df.groupby('arch')]
    df1['diff_score'] = df1['score'].values - df2['score'].values
    del df1['arch']
    del df1['score']
    return df1