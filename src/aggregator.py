import pandas as pd
import yaml
from itertools import count

from src import config

# TODO organize file-system like organization of methods? evaluation directory above arch directory? 


class Aggregator:
    def __init__(self, ev_name):
        self.ev_name = ev_name
        self.df_index = ['embed_size',
                         'time_of_init',
                         'embedder',
                         'arch',
                         'evaluation',
                         'task',
                         'replication',
                         'stage',
                         'shuffled_supervision',
                         'score']
        self.counter = count(0, 1)

    @staticmethod
    def load_param2val(embedder_p):
        with (embedder_p / 'params.yaml').open('r') as f:
            res = yaml.load(f)
        return res

    @staticmethod
    def get_embedder_name(param2val):
        for embedder_class in ['rnn', 'w2vec', 'count', 'random']:
            key = embedder_class + '_type'
            try:
                v = param2val[key]
            except KeyError:
                continue
            else:
                res = v if not isinstance(v, list) else v[0]
                return res
        else:
            raise RuntimeError('Did not find embedder name')

    def make_df(self):
        dfs = []
        for embedder_p in config.Dirs.runs.glob('*'):
            print('\n\n////////////////////////////////////////////////')
            print(embedder_p)
            param2val = self.load_param2val(embedder_p)
            #
            embed_size = param2val['embed_size'] if 'embed_size' in param2val else param2val['reduce_type'][1]
            time_of_init = embedder_p.name
            embedder = self.get_embedder_name(param2val)
            #
            df = self.make_embedder_df(embedder_p, embed_size, time_of_init, embedder)
            if len(df) > 0:
                dfs.append(df)
        res = pd.concat(dfs, axis=0)
        if dfs:
            return res
        else:
            return pd.DataFrame()

    def make_embedder_df(self, embedder_p, embed_size, time_of_init, embedder):
        dfs = []
        for arch_p in embedder_p.glob('*/{}'.format(self.ev_name)):
            arch = arch_p.parent.name
            print('\t', arch)
            print('\t\t', self.ev_name)
            df = self.make_arch_df(arch_p, embed_size, time_of_init, embedder, arch, self.ev_name)
            if len(df) > 0:
                dfs.append(df)
        if dfs:
            return pd.concat(dfs, axis=0)
        else:
            return pd.DataFrame()

    def make_arch_df(self, arch_p, embed_size, time_of_init, embedder, arch, ev):
        dfs = []
        for task_p in arch_p.glob('*/**'):
            task = task_p.name
            print('\t\t\t', task)
            df = self.make_task_df(task_p, embed_size, time_of_init, embedder, arch, ev, task)
            if len(df) > 0:
                dfs.append(df)
        if dfs:
            return pd.concat(dfs, axis=0)
        else:
            return pd.DataFrame()

    def make_task_df(self, task_p, embed_size, time_of_init, embedder, arch, ev, task):
        dfs = []
        for rep_p in task_p.glob('scores_*.csv'):
            rep = rep_p.name.split('.')[0][-1]  # TODO no more than 9 reps because of 1 digit
            print('\t\t\t\t', rep)
            df = self.make_rep_df(rep_p, embed_size, time_of_init, embedder, arch, ev, task, rep)
            if len(df) > 0:
                dfs.append(df)
        if dfs:
            return pd.concat(dfs, axis=0)
        else:
            return pd.DataFrame()

    def make_rep_df(self, rep_p, embed_size, time_of_init, embedder, arch, ev, task, rep):
        dfs = []
        scores_df = pd.read_csv(rep_p, index_col=False)
        for stage in ['exp', 'nov']:
            print('\t\t\t\t\t', stage)
            df = self.make_stage_df(scores_df, embed_size, time_of_init, embedder, arch, ev, task, rep, stage)
            if len(df) > 0:
                dfs.append(df)
        if dfs:
            return pd.concat(dfs, axis=0)
        else:
            return pd.DataFrame()

    def make_stage_df(self, scores_df, embed_size, time_of_init, embedder, arch, ev, task, rep, stage):
        dfs = []
        for shuffled in [True, False]:
            print('\t\t\t\t\t\t', shuffled)
            df = self.make_shuffled_df(scores_df, embed_size, time_of_init, embedder, arch, ev, task, rep, stage, shuffled)
            if len(df) > 0:
                print()
                dfs.append(df)
        if dfs:
            return pd.concat(dfs, axis=0)
        else:
            return pd.DataFrame()

    def make_shuffled_df(self, scores_df, embed_size, time_of_init, embedder, arch, ev, task, rep, stage, shuffled):
        vals = [embed_size, time_of_init, embedder, arch, ev, task, rep, stage, shuffled]
        # get score
        bool_id = scores_df['shuffled'] == shuffled
        score = scores_df[bool_id]['{}_score'.format(stage)].max()
        vals.append(score)
        #
        res = pd.DataFrame(index=[next(self.counter)], data={k: v for k, v in zip(self.df_index, vals)})
        return res

