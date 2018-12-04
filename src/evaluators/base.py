import pandas as pd
import multiprocessing as mp
import numpy as np

from src import config
from src.params import make_param2val_list, ObjectView


class ResultsData:
    def __init__(self, params_id, eval_candidates_mat):
        self.params_id = params_id
        self.eval_sims_mats = [np.full_like(eval_candidates_mat, np.nan, dtype=float)
                               for _ in range(config.Eval.num_evals)]


class Trial(object):
    def __init__(self, params_id, params):
        self.params_id = params_id
        self.params = params
        self.df_row = None
        self.results = None


class EvalBase(object):
    def __init__(self,
                 arch_name,
                 arch_params,
                 init_results_data,
                 split_and_vectorize_eval_data,
                 make_graph,
                 train_expert_on_train_fold,
                 train_expert_on_test_fold,
                 name, data_name1, data_name2, ev_params_class):
        # pass functions separately because module cannot be pickled (which multiprocessing requires)
        self.init_results_data = init_results_data
        self.split_and_vectorize_eval_data = split_and_vectorize_eval_data
        self.make_graph = make_graph
        self.train_expert_on_train_fold = train_expert_on_train_fold
        self.train_expert_on_test_fold = train_expert_on_test_fold
        #
        self.arch_name = arch_name
        self.name = name
        self.data_name1 = data_name1
        self.data_name2 = data_name2
        self.full_name = '{}_{}_{}_{}'.format(arch_name, self.name, data_name1, data_name2)
        #
        self.param2val_list = make_param2val_list(arch_params, ev_params_class)
        self.trials = [Trial(n, ObjectView(param2val))
                       for n, param2val in enumerate(self.param2val_list)]
        merged_keys = set(list(arch_params.__dict__.keys()) + list(ev_params_class.__dict__.keys()))
        self.df_header = sorted([k for k in merged_keys if not k.startswith('_')])
        #
        self.novice_score = None
        self.row_words = None
        self.col_words = None
        self.eval_candidates_mat = None

    # ////////////////////////////////////////////////////// evaluator-specific

    def make_all_eval_data(self, vocab_sims_mat, vocab):
        raise NotImplementedError('Must be implemented in child-class.')

    def check_negative_example(self, trial, p=None, c=None):
        raise NotImplementedError('Must be implemented in child-class.')

    def score(self, eval_sims_mat):
        raise NotImplementedError('Must be implemented in child-class.')

    def print_score(self, expert_score, is_expert):
        raise NotImplementedError('Must be implemented in child-class.')

    def to_eval_sims_mat(self, sims_mat):
        raise NotImplementedError('Must be implemented in child-class.')

    # //////////////////////////////////////////////////////

    def downsample(self, all_eval_probes, all_eval_candidates_mat, rep_id):
        # shuffle + down sample
        np.random.seed(rep_id)
        num_all = len(all_eval_candidates_mat)
        min_num_all = min(num_all, config.Eval.max_num_all) if self.name == 'matching' else num_all
        row_words = []
        eval_candidates_mat = []
        for rnd_id in np.random.choice(np.arange(min_num_all), size=min_num_all, replace=False):
            row_words.append(all_eval_probes[rnd_id])
            eval_candidates_mat.append(all_eval_candidates_mat[rnd_id])
        eval_candidates_mat = np.vstack(eval_candidates_mat)
        col_words = sorted(np.unique(eval_candidates_mat).tolist())
        return row_words, col_words, eval_candidates_mat

    # ////////////////////////////////////////////////////// train + score

    def score_novice(self, sims_mat):
        eval_sims_mat = self.to_eval_sims_mat(sims_mat)
        self.novice_score = self.score(eval_sims_mat)
        self.print_score(self.novice_score, is_expert=False)

    def train_and_score_expert(self, embedder, rep_id):
        # need to remove scores - this function is called only if replication is incomplete or config.retrain
        data_name = '{}_{}'.format(self.data_name1, self.data_name2)
        fname = 'scores_{}.csv'.format(rep_id)
        p = config.Dirs.runs / embedder.time_of_init / self.arch_name / self.name / data_name / fname
        if p.exists():
            print('Removing {}'.format(p))
            p.unlink()
        # run each trial in separate process
        pool = mp.Pool(processes=config.Eval.num_processes if not config.Eval.debug else 1)
        if config.Eval.debug:
            self.do_trial(self.trials[0], embedder.w2e, embedder.dim1)  # cannot pickle tensorflow errors
            raise SystemExit('Exited debugging mode successfully. Turn off debugging mode to train on all evaluators.')
        results = [pool.apply_async(self.do_trial, args=(trial, embedder.w2e, embedder.dim1))
                   for trial in self.trials]
        df_rows = []
        try:
            for res in results:
                df_row = res.get()
                df_rows.append(df_row)
        except KeyboardInterrupt:
            pool.close()
            raise SystemExit('Interrupt occurred during multiprocessing. Closed worker pool.')
        # save score obtained in each trial
        for df_row in df_rows:
            if config.Eval.save_scores:
                print('Saving trial score')
                df = pd.DataFrame(data=[df_row],
                                  columns=['exp_score', 'nov_score'] + self.df_header)
                print(df)
                if not p.parent.exists():
                    p.parent.mkdir(parents=True)
                with p.open('a') as f:
                    df.to_csv(f, mode='a', header=f.tell() == 0,
                              index=False)
        pool.close()

    def get_best_trial_score(self, trial):
        best_expert_score = 0
        best_eval_id = 0
        for eval_id, eval_sims_mat in enumerate(trial.results.eval_sims_mats):
            print('Eval {}'.format(eval_id + 1))
            expert_score = self.score(eval_sims_mat)
            self.print_score(expert_score, is_expert=True)
            if expert_score > best_expert_score:
                best_expert_score = expert_score
                best_eval_id = eval_id
        print('Expert score={:.2f} (at eval step {})'.format(best_expert_score, best_eval_id + 1))
        return best_expert_score

    def do_trial(self, trial, w2e, embed_size):
        trial.results = self.init_results_data(self, ResultsData(trial.params_id, self.eval_candidates_mat))
        assert hasattr(trial.results, 'params_id')
        print('Training expert on "{}"'.format(self.full_name))
        # train on each train-fold separately (fold_id is test_fold)
        for fold_id in range(config.Eval.num_folds):
            print('Fold {}/{}'.format(fold_id + 1, config.Eval.num_folds))
            data = self.split_and_vectorize_eval_data(self, trial, w2e, fold_id)
            graph = self.make_graph(self, trial, embed_size)
            self.train_expert_on_train_fold(self, trial, graph, data, fold_id)
            try:
                self.train_expert_on_test_fold(self, trial, graph, data, fold_id)  # TODO test
            except NotImplementedError:
                pass
        # score trial
        assert self.novice_score is not None
        df_row = [self.get_best_trial_score(trial), self.novice_score] + \
                       [trial.params.__dict__[p] for p in self.df_header]
        return df_row

    # ////////////////////////////////////////////////////// figs

    def make_trial_figs(self, trial):
        raise NotImplementedError('Must be implemented in child-class.')

    def save_figs(self, embedder):  # TODO test
        for trial in self.trials:
            for fig, fig_name in self.make_trial_figs(trial):
                trial_dname = 'trial_{}'.format(trial.params_id)
                fname = '{}_{}.png'.format(fig_name, trial.params_id)
                p = config.Dirs.runs / embedder.time_of_init / self.arch_name / self.name / trial_dname / fname
                if not p.parent.exists():
                    p.parent.mkdir(parents=True)
                fig.savefig(str(p))
                print('Saved {} to {}'.format(fig_name, p))