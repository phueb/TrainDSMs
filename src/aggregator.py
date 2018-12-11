import pandas as pd
import yaml
from itertools import count
import matplotlib.pyplot as plt
import copy
import numpy as np

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
                         'score']
        self.counter = count(0, 1)
        self.stages = ['novice', 'expert', 'expert+rs']

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
        for stage in self.stages:
            print('\t\t\t\t\t', stage)
            df = self.make_stage_df(scores_df, embed_size, time_of_init, embedder, arch, ev, task, rep, stage)
            if len(df) > 0:
                dfs.append(df)  # TODO try removing all tehse if statements
        if dfs:
            return pd.concat(dfs, axis=0)
        else:
            return pd.DataFrame()

    def make_stage_df(self, scores_df, embed_size, time_of_init, embedder, arch, ev, task, rep, stage):
        vals = [embed_size, time_of_init, embedder, arch, ev, task, rep, stage]
        if stage == self.stages[0]:
            bool_id = scores_df['shuffled'] == False
            score = scores_df[bool_id]['nov_score'].max()
            vals.append(score)
            return pd.DataFrame(index=[next(self.counter)], data={k: v for k, v in zip(self.df_index, vals)})
        elif stage == self.stages[1]:
            bool_id = scores_df['shuffled'] == False
            score = scores_df[bool_id]['exp_score'].max()
            vals.append(score)
            return pd.DataFrame(index=[next(self.counter)], data={k: v for k, v in zip(self.df_index, vals)})
        elif stage == self.stages[2]:
            bool_id = scores_df['shuffled'] == True
            score = scores_df[bool_id]['exp_score'].max()
            vals.append(score)
            return pd.DataFrame(index=[next(self.counter)], data={k: v for k, v in zip(self.df_index, vals)})

    # ///////////////////////////////////////////////////// plotting
    
    def show_task_plot(self,
                       arch_name,
                       task_name,  # TODO
                       embed_size,
                       include_dict=None,
                       min_num_reps=2,
                       y_step=0.1,
                       ax_fontsize=24,
                       t_fontsize=24,
                       dpi=192,
                       height=10,
                       width=10,
                       leg1_y=1.2):
        df = self.make_df()  # TODO use df
        #
        colors = plt.cm.get_cmap('tab10')
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        ylabel, ylims, yticks, y_chance = self.make_y_label_lims_ticks(y_step)
        title = 'Scores for {} + {}'.format(arch_name, self.ev_name)
        plt.title(title, fontsize=t_fontsize, y=leg1_y)
        # axis
        ax.yaxis.grid(True)
        ax.set_ylim(ylims)
        plt.ylabel(ylabel, fontsize=ax_fontsize)
        ax.set_xlabel(include_dict or [], fontsize=ax_fontsize)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_axisbelow(True)  # grid belw bars
        # plot
        ax.axhline(y=y_chance, color='grey', zorder=0)
        lines = []
        bw = 1.0  # TODO
        for embedder_id, (embedder_data, param2val) in enumerate(zip(embedders_data, param2val_list)):
            lines = []
            for stage in self.stages:
                x = 0.0  # TODO
                l1, = ax.bar(x + 0 * bw, y_e.mean(), width=bw, yerr=y_e.std(), color=colors(task_id))
                l2, = ax.bar(x + 1 * bw, y_s.mean(), width=bw, yerr=y_s.std(), color=colors(task_id), alpha=0.75)
                l3, = ax.bar(x + 3 * bw, y_n, width=bw, color=colors(task_id), alpha=0.5)
                lines.append((copy.copy(l1), copy.copy(l2), copy.copy(l3)))
        # tick labels
        ax.set_xticks([])  # TODO
        ax.set_xticklabels(['\n'.join(['{}: {}'.format(k, v) for k, v in param2val.items()
                                       if k not in include_dict.keys()])
                            for param2val in param2val_list],
                           fontsize=ax_fontsize)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=ax_fontsize)
        # legend

        plt.tight_layout()
        labels2 = self.stages
        self.add_double_legend(lines, labels1, labels2, leg1_y)  # TODO labels1 are embedders
        fig.subplots_adjust(bottom=0.1)
        plt.show()
        
    def make_y_label_lims_ticks(self, y_step):
        if self.ev_name == 'matching':
            ylabel = 'Balanced Accuracy'
            ylims = [0.5, 1]
            yticks = np.arange(0.5, 1, y_step).round(2)
            y_chance = 0.50
        elif self.ev_name == 'identification':
            ylabel = 'Accuracy'
            ylims = [0, 1]
            yticks = np.arange(0, 1 + y_step, y_step)
            y_chance = 0.20
        else:
            raise AttributeError('Invalid arg to "EVALUATOR_NAME".')
        return ylabel, ylims, yticks, y_chance

    @staticmethod
    def add_double_legend(lines_list, labels1, labels2, leg1_y, leg_fs=12, num_leg1_cols=8, num_leg2_cols=4):
        leg1 = plt.legend([l[0] for l in lines_list], labels1, loc='upper center',
                          bbox_to_anchor=(0.5, leg1_y), ncol=num_leg1_cols, frameon=False, fontsize=leg_fs)
        for lines in lines_list:
            for line in lines:
                line.set_color('black')
        plt.legend(lines_list[0], labels2, loc='upper center',
                   bbox_to_anchor=(0.5, leg1_y - 0.2), ncol=num_leg2_cols, frameon=False, fontsize=leg_fs)
        plt.gca().add_artist(leg1)  # order of legend creation matters here

