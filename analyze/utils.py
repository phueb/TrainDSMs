from itertools import product
from collections import Counter

from traindsms import config
from traindsms.aggregator import Aggregator


def gen_param2vals_for_completed_jobs(local=False):
    locations = config.Dirs.runs.glob('test/test') if local else config.RemoteDirs.runs.glob('**/*num*')
    for location in locations:
        param_name, job_name = location.parts[-2:]
        param2val = Aggregator.load_param2val(param_name, local)
        param2val['job_name'] = job_name
        yield param2val


def to_label(s):
    if s == 'nyms_syn_jw' or s == s == 'nyms_syn':
        return 'synonyms'
    elif s == 'nyms_ant_jw' or s == s == 'nyms_ant':
        return 'antonyms'
    elif s == 'cohyponyms_semantic':
        return 'cohyponyms'
    elif s == 'random_normal':
        return 'random'
    else:
        return s


def to_diff_df(df):
    df.drop(df[df['regime'].isin(['novice', 'control'])].index, inplace=True)
    df.drop(df[df['neg_pos_ratio'] == 0.0].index, inplace=True)
    df.drop(df[df['embedder'] == 'random_normal'].index, inplace=True)
    del df['corpus']
    del df['num_vocab']
    del df['embed_size']
    del df['evaluation']
    del df['param_name']
    del df['regime']
    del df['neg_pos_ratio']
    del df['num_epochs']
    #
    df1, df2 = [x for _, x in df.groupby('arch')]
    df1['diff_score'] = df1['score'].values - df2['score'].values
    del df1['arch']
    del df1['score']
    return df1


def check_duplicate_pairs(corpus_name, num_vocab):
    for p in config.Dirs.tasks.rglob('{}_{}*.txt'.format(corpus_name, num_vocab)):
        with p.open('r') as f:
            lines = f.read().splitlines()  # removes '\n' newline character
        unique_pairs = set()
        all_pairs = []
        for line in lines:
            probe = line.split()[0]
            relata = line.split()[1:]
            pairs_in_line = list(product([probe], relata))
            unique_pairs.update(pairs_in_line)
            all_pairs.extend(pairs_in_line)
        # check
        print()
        print(p.relative_to(config.Dirs.tasks))
        c = Counter(all_pairs)
        for pair, num in c.items():
            if num > 1:
                print(pair, num)


def make_task_name2_probe_data(corpus_name, num_vocab):
    res = {}
    for p in config.Dirs.tasks.rglob('{}_{}*.txt'.format(corpus_name, num_vocab)):
        with p.open('r') as f:
            lines = f.read().splitlines()  # removes '\n' newline character
        unique_pairs = set()
        num_pos_possible = 0
        probes = set()
        for line in lines:
            probe = line.split()[0]
            relata = line.split()[1:]
            pairs_in_line = list(product([probe], relata))
            unique_pairs.update(pairs_in_line)
            probes.update([probe] + relata)
            num_pos_possible += len(relata)
            # check
            if len(set(relata)) != len(relata):
                print('WARNING: Duplicate relata in line.')
        # task_name
        try:
            suffix = str(p.relative_to(config.Dirs.tasks).stem).split('_')[2]
        except IndexError:
            suffix = ''
        task_name = '{}{}'.format(str(p.relative_to(config.Dirs.tasks).parent).replace('/', '_'),
                                  '_' + suffix if suffix else '')
        #
        num_row_words = len(lines)
        num_unique_probes = len(probes)
        num_total_possible = len(probes) ** 2
        num_pos = len(unique_pairs)
        num_neg = num_total_possible - num_pos
        diff = num_pos_possible - num_pos
        res[task_name] = \
            (num_row_words, num_unique_probes, num_total_possible, num_pos, num_neg, num_pos_possible, diff)
    return res