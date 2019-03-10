import pandas as pd
import yaml
from itertools import count, cycle
import matplotlib.pyplot as plt
import copy
import numpy as np
import datetime
import time

from two_process_nlp import config
from two_process_nlp.params import to_embedder_name


class Aggregator:
    def __init__(self):
        self.expert_param_names = ['neg_pos_ratio', 'standardize', 'num_epochs']  # TODo test standardize
        self.df_index = ['corpus',
                         'num_vocab',
                         'embed_size',
                         'param_name',
                         'job_name',
                         'embedder',
                         'arch',
                         'evaluation',
                         'task',
                         'regime'] + self.expert_param_names + ['score']
        self.df_name_with_date = '2process_data_{}.csv'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        self.df_name = '2process_data.csv'
        self.df = None
        self.counter = count(0, 1)
        # constants
        self.bw = 0.2
        self.hatches = ['--', '\\\\', 'xx']
        self.embedder2color = {embedder_name: plt.cm.get_cmap('tab10')(n) for n, embedder_name in enumerate(
            ['ww', 'wd', 'sg', 'cbow', 'srn', 'lstm', 'random_normal', 'random_uniform', 'glove'])}

    @classmethod
    def load_param2val(cls, param_name):
        with (config.RemoteDirs.runs / param_name / 'param2val.yaml').open('r') as f:
            res = yaml.load(f)
        return res

    def make_df(self, load_from_file, verbose):
        # load from file
        p = config.RemoteDirs.root / self.df_name
        if p.exists() and load_from_file:
            print('Loading data frame from file. Re-export data to file if data has changed')
            res = pd.read_csv(p)
            self.df = res
            return res
        # make from runs data
        dfs = []
        for location in config.RemoteDirs.runs.glob('**/*num*'):
            if verbose:
                print()
                print(location)
            param_name, job_name = location.parts[-2:]
            param2val = self.load_param2val(param_name)
            #
            corpus = param2val['corpus_name']
            num_vocab = param2val['num_vocab']
            embed_size = param2val['embed_size'] if 'embed_size' in param2val else param2val['reduce_type'][1]
            embedder = to_embedder_name(param2val)
            #
            df = self.make_embedder_df(corpus, num_vocab, embed_size, param_name, job_name, embedder, verbose)
            if len(df) > 0:
                dfs.append(df)
        if dfs:
            res = pd.concat(dfs, axis=0)
            self.df = res
            print('Num rows in df={}'.format(len(res)))
            return res
        else:
            raise RuntimeError('Did not find any scores in {}'.format(config.RemoteDirs.runs))

    def make_embedder_df(self, corpus, num_vocab, embed_size, param_name, job_name, embedder, verbose):
        dfs = []
        loc = config.RemoteDirs.runs / param_name / job_name
        for scores_p in loc.rglob('scores.csv'):
            arch, ev, task, regime = scores_p.relative_to(loc).parts[:-1]
            scores_df = pd.read_csv(scores_p, index_col=False)
            # group by expert_params which are present in scores_df
            group_names = [pn for pn in self.expert_param_names
                                            if pn in scores_df.columns] + ['score']
            # collect score for each row in scores_df
            for param_vals_and_score, group in scores_df.groupby(group_names):
                it = iter([pn if pn != 'None' else np.nan for pn in param_vals_and_score])  # None to nan
                all_param_vals_and_score = [next(it) if pn in group_names else np.nan
                                            for pn in self.expert_param_names + ['score']]
                vals = [corpus, num_vocab, embed_size, param_name, job_name,
                        embedder, arch, ev, task, regime] + all_param_vals_and_score
                if verbose:
                    for n, v in enumerate(vals[5:]):
                        print('\t' * (n + 1), v)

                # TODO add num_epochs column


                df = pd.DataFrame(index=[next(self.counter)], data={k: v for k, v in zip(self.df_index, vals)})
                dfs.append(df)
        if dfs:
            return pd.concat(dfs, axis=0)
        else:
            return pd.DataFrame()

    # ///////////////////////////////////////////////////// plotting

    def make_task_plot(self,
                       corpus,
                       num_vocab,
                       arch,
                       ev,
                       task,
                       embed_size,
                       neg_pos_ratio,
                       num_epochs,
                       load_from_file=False,
                       verbose=True,
                       save=False,
                       min_num_reps=2,
                       y_step=0.1,
                       xax_fontsize=6,
                       yax_fontsize=20,
                       t_fontsize=20,
                       dpi=192,
                       height=8,
                       width=14,
                       leg1_y=1.2):
        # filter by arch + task + embed_size + evaluation
        df = self.make_df(load_from_file, verbose)
        bool_id = (df['arch'] == arch) & \
                  (df['task'] == task) & \
                  (df['embed_size'] == embed_size) & \
                  (df['neg_pos_ratio'].isin([np.nan, neg_pos_ratio])) & \
                  (df['num_epochs'].isin([np.nan, num_epochs])) & \
                  (df['evaluation'] == ev) & \
                  (df['corpus'] == corpus) & \
                  (df['num_vocab'] == num_vocab)
        filtered_df = df[bool_id]
        # fig
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        ylabel, ylims, yticks, y_chance = self.make_y_label_lims_ticks(y_step, ev)
        title = 'Scores for\n{} + {} + {} + embed_size={} + neg_pos_ratio={} + num_epochs={}\n' \
                '{} num_vocab={}'.format(arch, ev, task, embed_size, neg_pos_ratio, num_epochs,
                                         corpus, num_vocab)
        plt.title(title, fontsize=t_fontsize, y=leg1_y)
        # axis
        ax.yaxis.grid(True)
        ax.set_ylim(ylims)
        plt.ylabel(ylabel, fontsize=yax_fontsize)
        ax.set_xlabel(None)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_axisbelow(True)  # set grid below bars
        # plot
        ax.axhline(y=y_chance, color='grey', zorder=0)
        bars_list = []
        param2val_list = []
        embedder_names = []
        novice_df = filtered_df[filtered_df['regime'] == 'novice']
        param_names_sorted_by_score = novice_df.groupby('param_name').mean().sort_values(
            'score', ascending=False).index.values
        hatches = None
        regimes = None
        for param_id, param_name in enumerate(param_names_sorted_by_score):
            #
            bool_id = df['param_name'] == param_name
            embedder_df = filtered_df[bool_id]
            #
            param2val = self.load_param2val(param_name)
            param2val_list.append(param2val)
            embedder_name = to_embedder_name(param2val)
            #
            print()
            print(param_name)
            print(embedder_name)
            print('num_scores={}'.format(len(embedder_df)))
            # hatches
            hatches = self.hatches.copy()
            grouped = embedder_df.groupby('regime')
            regimes = [p for p, g in grouped]
            if 'expert' not in embedder_df['regime'].unique():
                print(('WARNING: Found only regime(s): {}'.format(regimes)))
                hatches.pop()
            if 'control' not in embedder_df['regime'].unique():
                print(('WARNING: Found only regime(s): {}'.format(regimes)))
                hatches.pop()
            hatches = cycle(hatches)
            # bars
            bars = []
            x = param_id + 0.6
            for regime, regime_df in grouped:
                ys = regime_df['score'].values
                print(ys)
                if len(ys) < min_num_reps:
                    print('Skipping due to num_reps={}<min_num_reps'.format(len(ys)))
                    continue
                x += self.bw
                ys = regime_df['score']
                print('{:<10} score mean={:.2f} std={:.3f} n={:>2}'.format(regime, ys.mean(), ys.std(), len(ys)))
                b, = ax.bar(x + 0 * self.bw, ys.mean(),
                            width=self.bw,
                            yerr=ys.std(),
                            color=self.embedder2color[embedder_name],
                            edgecolor='black',
                            hatch=next(hatches))
                bars.append(copy.copy(b))
            if bars:
                bars_list.append(bars)
                embedder_names.append(embedder_name)
        print('Found {} embedders.'.format(len(bars_list)))
        # tick labels
        num_embedders = len(param2val_list)
        ax.set_xticks(np.arange(1, num_embedders + 1, 1))
        hidden_keys = ['count_type', 'corpus_name']
        excluded_keys = ['num_vocab', 'corpus_name', 'embed_size', 'job_name', 'param_name']
        ax.set_xticklabels(['\n'.join(['{}{}'.format(k + ': ' if k not in hidden_keys else '', v)
                                       for k, v in param2val.items()
                                       if k not in excluded_keys])
                            for param2val in param2val_list],
                           fontsize=xax_fontsize)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=yax_fontsize)
        # legend
        plt.tight_layout()
        labels1 = embedder_names
        labels2 = regimes
        if not bars_list:
            print('WARNING:No scores found for given factors.')
            return
        self.add_double_legend(bars_list, labels1, labels2, leg1_y, num_embedders, hatches)
        fig.subplots_adjust(bottom=0.1)
        if not save:
            plt.show()
        else:
            time_of_fig = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            p = config.LocalDirs.root / 'figs' / '{}.png'.format(time_of_fig)
            print('Saving fig to {}'.format(p))
            plt.savefig(p.open('wb'), bbox_inches="tight")
            time.sleep(1)

    @staticmethod
    def make_y_label_lims_ticks(y_step, eval):
        if eval == 'matching':
            ylabel = 'Balanced Accuracy'
            ylims = [0.5, 1]
            yticks = np.arange(0.5, 1, y_step).round(2)
            y_chance = 0.50
        elif eval == 'identification':
            ylabel = 'Accuracy'
            ylims = [0, 1]
            yticks = np.arange(0, 1 + y_step, y_step).round(2)
            y_chance = 0.0  # TODO dynamic
        else:
            raise AttributeError('Invalid arg to "EVALUATOR_NAME".')
        return ylabel, ylims, yticks, y_chance

    def add_double_legend(self, bars_list, labels1, labels2, leg1_y, num_leg1_cols, hatches,
                          leg_fs=12, num_leg2_cols=4):
        for bars in bars_list:
            for bar in bars:
                bar.set_hatch(None)
        leg1 = plt.legend([bar[0] for bar in bars_list], labels1, loc='upper center',
                          bbox_to_anchor=(0.5, leg1_y), ncol=num_leg1_cols, frameon=False, fontsize=leg_fs)
        for bars in bars_list:
            for bar in bars:
                bar.set_facecolor('white')
                bar.set_hatch(next(hatches))
        plt.legend(bars_list[0], labels2, loc='upper center',
                   bbox_to_anchor=(0.5, leg1_y - 0.1), ncol=num_leg2_cols, frameon=False, fontsize=leg_fs)
        plt.gca().add_artist(leg1)  # order of legend creation matters here
