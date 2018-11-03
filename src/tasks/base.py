import pandas as pd
import multiprocessing as mp

from src import config
from src.params import make_param2id, ObjectView


class EvalData:
    def __init__(self, params_id):
        self.params_id = params_id


class Trial(object):
    def __init__(self, params_id, params):
        self.params_id = params_id
        self.params = params
        self.df_row = None
        self.eval = None


class TaskBase(object):
    def __init__(self, name, Params):
        self.name = name
        self.param2val_list = list(make_param2id(Params, stage1=False))
        self.trials = [Trial(n, ObjectView(param2val))
                       for n, param2val in enumerate(self.param2val_list)]
        self.df_header = sorted([k for k in Params.__dict__.keys() if not k.startswith('_')])
        self.novice_score = None

    def init_eval_data(self, trial):
        return EvalData(trial.params_id)

    def make_data(self, trial, w2e, fold_id):
        raise NotImplementedError('Must be implemented in child-class.')

    def make_graph(self, trial, embed_size):
        raise NotImplementedError('Must be implemented in child-class.')

    def train_expert_on_train_fold(self, trial, graph, data, fold_id):
        raise NotImplementedError('Must be implemented in child-class.')

    def train_expert_on_test_fold(self, trial, graph, data, fold_id):
        raise NotImplementedError('Must be implemented in child-class.')

    def get_best_trial_score(self, trial):
        raise NotImplementedError('Must be implemented in child-class.')

    # ////////////////////////////////////////////////////// train + score

    def do_trial(self, trial, embedder):
        trial.eval = self.init_eval_data(trial)
        assert hasattr(trial.eval, 'params_id')
        print('Training {} expert'.format(self.name))
        # train on each train-fold separately (fold_id is test_fold)
        for fold_id in range(config.Task.num_folds):
            print('Fold {}/{}'.format(fold_id + 1, config.Task.num_folds))
            graph = self.make_graph(trial, embedder.dim1)
            data = self.make_data(trial, embedder.w2e, fold_id)
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
        pool = mp.Pool(processes=config.Task.num_processes)
        results = [pool.apply_async(self.do_trial, args=(trial, embedder)) for trial in self.trials]
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
            if config.Task.save_scores:
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