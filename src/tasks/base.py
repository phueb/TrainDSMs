import pandas as pd
import numpy as np

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

    def train_expert_on_test_fold(self, trial, graph, data, fold_id):  # TODO
        raise NotImplementedError('Must be implemented in child-class.')

    def get_best_trial_score(self, trial):
        raise NotImplementedError('Must be implemented in child-class.')

    def train_and_score_expert(self, embedder):
        # remove scores
        p = config.Dirs.runs / embedder.time_of_init / self.name / 'scores.csv'
        if config.Task.clear_scores and p.exists():
            print('Removing previous scores')
            p.unlink()
        # train + score
        for trial in self.trials:
            trial.eval = self.init_eval_data(trial)
            assert hasattr(trial.eval, 'params_id')
            print('Training {} expert'.format(self.name))
            # train on folds
            for fold_id in range(trial.params.num_folds):
                print('Fold {}/{}'.format(fold_id + 1, trial.params.num_folds))
                graph = self.make_graph(trial, embedder.dim1)
                data = self.make_data(trial, embedder.w2e, fold_id)
                self.train_expert_on_train_fold(trial, graph, data, fold_id)
                try:
                    self.train_expert_on_test_fold(trial, graph, data, fold_id)  # TODO
                except NotImplementedError:
                    pass
            # score trial
            assert self.novice_score is not None
            trial.df_row = [self.get_best_trial_score(trial), self.novice_score] + \
                           [trial.params.__dict__[p] for p in self.df_header]
            # save
            if config.Task.save_scores:
                print('Saving trial score')
                df = pd.DataFrame(data=[trial.df_row],
                                   columns=['exp_score', 'nov_score'] + self.df_header)
                print(df)
                if not p.parent.exists():
                    p.parent.mkdir()
                with p.open('a') as f:
                    df.to_csv(f, mode='a', header=f.tell() == 0, index=False)

    def save_figs(self, embedder, make_figs):  # TODO this function is called from child class via super() and passed make_figs
        for trial in self.trials:
            # figs
            for fig, fig_name in make_figs():
                p = config.Dirs.runs / embedder.time_of_init / self.name / '{}_{}.png'.format(fig_name, trial.params_id)
                if not p.parent.exists():
                    p.parent.mkdir(parents=True)
                fig.savefig(str(p))
                print('Saved {} to {}'.format(fig_name, p))