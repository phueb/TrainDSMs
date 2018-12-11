import pandas as pd
import yaml
from itertools import count, cycle
import matplotlib.pyplot as plt
import copy
import numpy as np
from pathlib import Path

from src import config

# TODO organize file-system like organization of methods? evaluation directory above arch directory? 


class Aggregator:
    def __init__(self, ev_name):
        self.ev = ev_name
        self.df_index = ['corpus_name',
                         'num_vocab',
                         'embed_size',
                         'time_of_init',
                         'embedder',
                         'arch',
                         'evaluation',
                         'task',
                         'replication',
                         'stage',
                         'score']
        self.df = None
        self.counter = count(0, 1)
        self.stages = ['expert', 'expert+rs', 'novice']

    @staticmethod
    def load_param2val(embedder_p=None, time_of_init=None):
        if time_of_init is not None:
            embedder_p = config.Dirs.runs / time_of_init
        if embedder_p is not None:
            with (embedder_p / 'params.yaml').open('r') as f:
                res = yaml.load(f)
        else:
            raise RuntimeError('Must specify "embedder_p" or "time_of_init" to retrieve param2val.')
        return res
    
    @staticmethod
    def to_embedder_name(param2val):
        if 'random_type' in param2val:
            return param2val['random_type']
        elif 'rnn_type' in param2val:
            return param2val['rnn_type']
        elif 'w2vec_type' in param2val:
            return param2val['w2vec_type']
        elif 'count_type' in param2val:
            return param2val['count_type'][0]
        else:
            raise RuntimeError('Unknown embedder name')

    def make_df(self, load_from_file=False):
        # load from file
        p = Path('{}.csv'.format(self.ev))
        if p.exists() and load_from_file:
            print('Loading data frame from file. Re-export data to file if data has changed')
            res = pd.read_csv(p)
            self.df = res
            return res
        # make from runs data
        dfs = []
        for embedder_p in config.Dirs.runs.glob('*'):
            print('\n\n////////////////////////////////////////////////')
            print(embedder_p)
            param2val = self.load_param2val(embedder_p)
            #
            corpus_name = param2val['corpus_name']
            num_vocab = param2val['num_vocab']
            embed_size = param2val['embed_size'] if 'embed_size' in param2val else param2val['reduce_type'][1]
            time_of_init = embedder_p.name
            embedder = self.to_embedder_name(param2val)
            #
            df = self.make_embedder_df(embedder_p, corpus_name, num_vocab, embed_size, time_of_init, embedder)
            if len(df) > 0:
                dfs.append(df)
        res = pd.concat(dfs, axis=0)
        if dfs:
            self.df = res
            return res
        else:
            return pd.DataFrame()

    def make_embedder_df(self, embedder_p, corpus_name, num_vocab, embed_size, time_of_init, embedder):
        dfs = []
        for arch_p in embedder_p.glob('*/{}'.format(self.ev)):
            arch = arch_p.parent.name
            print('\t', arch)
            print('\t\t', self.ev)
            df = self.make_arch_df(arch_p, corpus_name, num_vocab, embed_size, time_of_init, embedder, arch, self.ev)
            if len(df) > 0:
                dfs.append(df)
        if dfs:
            return pd.concat(dfs, axis=0)
        else:
            return pd.DataFrame()

    def make_arch_df(self, arch_p, corpus_name, num_vocab, embed_size, time_of_init, embedder, arch, ev):
        dfs = []
        for task_p in arch_p.glob('*/**'):
            task = task_p.name
            print('\t\t\t', task)
            df = self.make_task_df(task_p, corpus_name, num_vocab, embed_size, time_of_init, embedder, arch, ev, task)
            if len(df) > 0:
                dfs.append(df)
        if dfs:
            return pd.concat(dfs, axis=0)
        else:
            return pd.DataFrame()

    def make_task_df(self, task_p, corpus_name, num_vocab, embed_size, time_of_init, embedder, arch, ev, task):
        dfs = []
        for rep_p in task_p.glob('scores_*.csv'):
            rep = rep_p.name.split('.')[0][-1]  # TODO no more than 9 reps because of 1 digit
            print('\t\t\t\t', rep)
            df = self.make_rep_df(
                rep_p, corpus_name, num_vocab, embed_size, time_of_init, embedder, arch, ev, task, rep)
            if len(df) > 0:
                dfs.append(df)
        if dfs:
            return pd.concat(dfs, axis=0)
        else:
            return pd.DataFrame()

    def make_rep_df(self, rep_p, corpus_name, num_vocab, embed_size, time_of_init, embedder, arch, ev, task, rep):
        dfs = []
        scores_df = pd.read_csv(rep_p, index_col=False)
        for stage in self.stages:
            print('\t\t\t\t\t', stage)
            df = self.make_stage_df(
                scores_df, corpus_name, num_vocab, embed_size, time_of_init, embedder, arch, ev, task, rep, stage)
            if len(df) > 0:
                dfs.append(df)
        if dfs:
            return pd.concat(dfs, axis=0)
        else:
            return pd.DataFrame()

    def make_stage_df(self,
                      scores_df,
                      corpus_name,
                      num_vocab,
                      embed_size,
                      time_of_init,
                      embedder,
                      arch,
                      ev,
                      task,
                      rep,
                      stage):
        vals = [corpus_name, num_vocab, embed_size, time_of_init, embedder, arch, ev, task, rep, stage]
        if stage == self.stages[0]:
            bool_id = scores_df['shuffled'] == False
            score = scores_df[bool_id]['exp_score'].max()
            vals.append(score)
            return pd.DataFrame(index=[next(self.counter)], data={k: v for k, v in zip(self.df_index, vals)})
        elif stage == self.stages[1]:
            bool_id = scores_df['shuffled'] == True
            score = scores_df[bool_id]['exp_score'].max()
            vals.append(score)
            return pd.DataFrame(index=[next(self.counter)], data={k: v for k, v in zip(self.df_index, vals)})
        elif stage == self.stages[2]:
            bool_id = scores_df['shuffled'] == False
            score = scores_df[bool_id]['nov_score'].max()
            vals.append(score)
            return pd.DataFrame(index=[next(self.counter)], data={k: v for k, v in zip(self.df_index, vals)})

    # ///////////////////////////////////////////////////// plotting
    
    def show_task_plot(self,
                       arch,
                       task,
                       embed_size,
                       load_from_file=False,
                       include_dict=None,
                       min_num_reps=2,
                       y_step=0.1,
                       xax_fontsize=6,
                       yax_fontsize=10,
                       t_fontsize=20,
                       dpi=192,
                       height=8,
                       width=14,
                       leg1_y=1.2):
        if include_dict is None:
            include_dict = {}
        # filter by arch + task + embed_size + evaluation
        df = self.make_df(load_from_file=load_from_file)  # TODO put corpus_name and num_vocab in df
        bool_id = (df['arch'] == arch) &\
                  (df['task'] == task) &\
                  (df['embed_size'] == embed_size) &\
                  (df['evaluation'] == self.ev)
        filtered_df = df[bool_id]
        # constants
        bw = 0.2
        hatches = cycle(['-', '\\', 'x'])
        colors = plt.cm.get_cmap('tab10')
        # fig
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        ylabel, ylims, yticks, y_chance = self.make_y_label_lims_ticks(y_step)
        title = 'Scores for\n{} + {} + {} + embed_size={}'.format(arch, self.ev, task, embed_size)
        plt.title(title, fontsize=t_fontsize, y=leg1_y)
        # axis
        ax.yaxis.grid(True)
        ax.set_ylim(ylims)
        plt.ylabel(ylabel, fontsize=yax_fontsize)
        ax.set_xlabel(include_dict or '', fontsize=xax_fontsize)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_axisbelow(True)  # grid belw bars
        # plot
        ax.axhline(y=y_chance, color='grey', zorder=0)
        bars_list = []
        param2val_list = []
        embedder_names = []
        for embedder_id, (embedder, embedder_df) in enumerate(filtered_df.groupby('embedder')):
            time_of_init = embedder_df['time_of_init'].iloc[0]
            param2val = self.load_param2val(time_of_init=time_of_init)
            param2val_list.append(param2val)
            embedder_name = self.to_embedder_name(param2val)
            embedder_names.append(embedder_name)
            #
            bars = []
            x = embedder_id + 0.6
            for stage, stage_df in embedder_df.groupby('stage'):  # gives df with len = num_reps
                num_reps = len(stage_df)
                if num_reps < min_num_reps:
                    print('Skipping due to num_reps={}<min_num_reps'.format(num_reps))
                    continue
                x += bw
                ys = stage_df['score']
                print(stage, ys.mean())
                b, = ax.bar(x + 0 * bw, ys.mean(),
                            width=bw,
                            yerr=ys.std(),
                            color=colors(embedder_id),
                            edgecolor='black',
                            hatch=next(hatches))
                bars.append(copy.copy(b))
            bars_list.append(bars)
        # tick labels
        num_embedders = len(param2val_list)
        ax.set_xticks(np.arange(1, num_embedders + 1, 1))
        ax.set_xticklabels(['\n'.join(['{}: {}'.format(k, v) for k, v in param2val.items()
                                       if k not in include_dict.keys()])
                            for param2val in param2val_list],
                           fontsize=xax_fontsize)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=yax_fontsize)
        # legend
        plt.tight_layout()
        labels1 = embedder_names
        labels2 = self.stages
        self.add_double_legend(bars_list, labels1, labels2, leg1_y)  # TODO show hatching correctly in legend
        fig.subplots_adjust(bottom=0.1)
        plt.show()
        
    def make_y_label_lims_ticks(self, y_step):
        if self.ev == 'matching':
            ylabel = 'Balanced Accuracy'
            ylims = [0.5, 1]
            yticks = np.arange(0.5, 1, y_step).round(2)
            y_chance = 0.50
        elif self.ev == 'identification':
            ylabel = 'Accuracy'
            ylims = [0, 1]
            yticks = np.arange(0, 1 + y_step, y_step)
            y_chance = 0.20
        else:
            raise AttributeError('Invalid arg to "EVALUATOR_NAME".')
        return ylabel, ylims, yticks, y_chance

    @staticmethod
    def add_double_legend(bars_list, labels1, labels2, leg1_y, leg_fs=12, num_leg1_cols=8, num_leg2_cols=4):
        leg1 = plt.legend([bar[0] for bar in bars_list], labels1, loc='upper center',
                          bbox_to_anchor=(0.5, leg1_y), ncol=num_leg1_cols, frameon=False, fontsize=leg_fs)
        for bars in bars_list:
            for bar in bars:
                bar.set_color('black')
        plt.legend(bars_list[0], labels2, loc='upper center',
                   bbox_to_anchor=(0.5, leg1_y - 0.1), ncol=num_leg2_cols, frameon=False, fontsize=leg_fs)
        plt.gca().add_artist(leg1)  # order of legend creation matters here