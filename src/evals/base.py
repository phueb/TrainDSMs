import pandas as pd
import multiprocessing as mp
import numpy as np

from src import config
from src.params import make_param2id, ObjectView


class ResultsData:
    def __init__(self, params_id, eval_candidates_mat):
        self.params_id = params_id
        self.eval_sims_mats = [np.zeros_like(eval_candidates_mat, float)  # TODO test
                               for _ in range(config.Eval.num_evals)]


class Trial(object):
    def __init__(self, params_id, params):
        self.params_id = params_id
        self.params = params
        self.df_row = None
        self.results = None


class EvalBase(object):
    def __init__(self, name, Params):
        self.name = name
        self.param2val_list = list(make_param2id(Params, stage1=False))
        self.trials = [Trial(n, ObjectView(param2val))
                       for n, param2val in enumerate(self.param2val_list)]
        self.df_header = sorted([k for k in Params.__dict__.keys() if not k.startswith('_')])
        #
        self.metric = None
        self.novice_score = None
        self.row_words = None
        self.col_words = None
        self.eval_candidates_mat = None

    # ////////////////////////////////////////////////////// eval-specific

    def to_eval_sims_mat(self, sims_mat):
        raise NotImplementedError('Must be implemented in child-class.')

    def make_eval_data(self, sims):
        raise NotImplementedError('Must be implemented in child-class.')

    def check_negative_example(self, trial, p=None, c=None):
        raise NotImplementedError('Must be implemented in child-class.')

    def score(self, eval_sims_mat, is_expert):
        raise NotImplementedError('Must be implemented in child-class.')

    # ////////////////////////////////////////////////////// eval-specific

    def init_results_data(self, trial):
        print('Initializing evaluation data structure')
        return ResultsData(trial.params_id, self.eval_candidates_mat)

    def split_and_vectorize_eval_data(self, trial, w2e, fold_id):
        raise NotImplementedError('Must be implemented in child-class.')

    def make_graph(self, trial, embed_size):
        raise NotImplementedError('Must be implemented in child-class.')

    def train_expert_on_train_fold(self, trial, graph, data, fold_id):
        raise NotImplementedError('Must be implemented in child-class.')

    def train_expert_on_test_fold(self, trial, graph, data, fold_id):
        raise NotImplementedError('Must be implemented in child-class.')

    # ////////////////////////////////////////////////////// train + score

    def score_novice(self, sims_mat):
        eval_sims_mat = self.to_eval_sims_mat(sims_mat)
        self.novice_score = self.score(eval_sims_mat, is_expert=False)

    def do_trial(self, trial, w2e, embed_size):
        trial.results = self.init_results_data(trial)
        assert hasattr(trial.results, 'params_id')
        print('Training {} expert'.format(self.name))
        # train on each train-fold separately (fold_id is test_fold)
        for fold_id in range(config.Eval.num_folds):
            print('Fold {}/{}'.format(fold_id + 1, config.Eval.num_folds))
            data = self.split_and_vectorize_eval_data(trial, w2e, fold_id)
            graph = self.make_graph(trial, embed_size)
            self.train_expert_on_train_fold(trial, graph, data, fold_id)
            try:
                self.train_expert_on_test_fold(trial, graph, data, fold_id)  # TODO test
            except NotImplementedError:
                pass
        # score trial
        assert self.novice_score is not None
        df_row = [self.get_best_trial_score(trial), self.novice_score] + \
                       [trial.params.__dict__[p] for p in self.df_header]
        return df_row

    def train_and_score_expert(self, embedder, rep_id):
        # need to remove scores - this function is called only if replication is incomplete or config.retrain
        p = config.Dirs.runs / embedder.time_of_init / self.name / 'scores_{}.csv'.format(rep_id)
        if p.exists():
            print('Removing {}'.format(p))
            p.unlink()
        # run each trial in separate process
        pool = mp.Pool(processes=config.Eval.num_processes if not config.Eval.debug else 1)
        if config.Eval.debug:
            self.do_trial(self.trials[0], embedder.w2e, embedder.dim1)  # cannot pickle tensorflow errors
            raise SystemExit('Exited debugging mode successfully. Turn off debugging mode to train on all evals.')
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
            if config.Eval.save_scores:  # TODO take into account architecture when saving data
                print('Saving trial score')
                df = pd.DataFrame(data=[df_row],
                                  columns=['exp_score', 'nov_score'] + self.df_header)
                print(df)
                if not p.parent.exists():
                    p.parent.mkdir()
                with p.open('a') as f:
                    df.to_csv(f, mode='a', header=f.tell() == 0,
                              index=False)
        # close pool
        pool.close()

    def get_best_trial_score(self, trial):
        best_expert_score = 0
        best_eval_id = 0
        for eval_id, eval_sims_mat in enumerate(trial.results.eval_sims_mats):
            expert_score = self.score(eval_sims_mat, is_expert=True)
            print('{} at eval {} is {:.2f}'.format(self.metric, eval_id + 1, expert_score))
            if expert_score > best_expert_score:
                best_expert_score = expert_score
                best_eval_id = eval_id
        print('Expert score={:.2f} (at eval step {})'.format(best_expert_score, best_eval_id + 1))
        return best_expert_score

    # ////////////////////////////////////////////////////// figs

    def make_trial_figs(self, trial):
        raise NotImplementedError('Must be implemented in child-class.')

    def save_figs(self, embedder):  # TODO test
        for trial in self.trials:
            for fig, fig_name in self.make_trial_figs(trial):
                trial_dname = 'trial_{}'.format(trial.params_id)
                fname = '{}_{}.png'.format(fig_name, trial.params_id)
                p = config.Dirs.runs / embedder.time_of_init / self.name / trial_dname / fname
                if not p.parent.exists():
                    p.parent.mkdir(parents=True)
                fig.savefig(str(p))
                print('Saved {} to {}'.format(fig_name, p))