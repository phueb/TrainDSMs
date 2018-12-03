import numpy as np
from functools import partial

from src import config
from src.scores import calc_balanced_accuracy
from src.figs import make_matching_figs
from src.evaluators.base import EvalBase


class MatchingParams:
    prop_negative = [0.3]  # 0.3 is better than 0.1 or 0.2 but not 0.5
    # arch-evaluator interaction
    num_epochs = [100]  # 100 is better than 10 or 20 or 50

    # TODO add option to balance neg:pos
    if any([i > 0.5 for i in prop_negative]):  # TODO test
        raise ValueError('Setting "prop_negative" too high might cause memory error.')


class Matching(EvalBase):
    def __init__(self, arch, data_name1, data_name2=None):
        super().__init__(arch, 'matching', data_name1, data_name2, MatchingParams)
        #
        probes, probe_relata = self.load_probes()  # relata can be synonyms, hypernyms, etc.
        relata = sorted(np.unique(np.concatenate(probe_relata)).tolist())
        self.probe2relata = {p: r for p, r in zip(probes, probe_relata)}
        # sims
        self.all_row_words = probes
        self.all_col_words = relata
        # misc
        self.binomial = np.random.binomial
        self.metric = config.Eval.matching_metric

    # ///////////////////////////////////////////// Overwritten Methods START

    def make_all_eval_data(self, vocab_sims_mat, vocab):
        num_row_words = len(self.all_row_words)
        all_eval_candidates_mat = np.asarray([self.all_col_words for _ in range(num_row_words)])
        return all_eval_candidates_mat

    def check_negative_example(self, trial, p=None, c=None):
        return bool(self.binomial(n=1, p=trial.params.prop_negative, size=1))

    def score(self, eval_sims_mat):
        # make gold (signal detection masks)
        num_rows = len(self.row_words)
        num_cols = len(self.col_words)
        res = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            relata1 = self.probe2relata[self.row_words[i]]
            for j in range(num_cols):
                relatum2 = self.col_words[j]
                if relatum2 in relata1:
                    res[i, j] = 1
        gold = res.astype(np.bool)

        def calc_signals(probe_sims, gold, thr):  # vectorized algorithm is 20X faster
            predicted = np.zeros_like(probe_sims, int)
            predicted[np.where(probe_sims > thr)] = 1
            tp = float(len(np.where((predicted == gold) & (gold == 1))[0]))
            tn = float(len(np.where((predicted == gold) & (gold == 0))[0]))
            fp = float(len(np.where((predicted != gold) & (gold == 0))[0]))
            fn = float(len(np.where((predicted != gold) & (gold == 1))[0]))
            return tp, tn, fp, fn

        # balanced acc
        calc_signals = partial(calc_signals, eval_sims_mat, gold)
        sims_mean = np.asscalar(np.mean(eval_sims_mat))
        res = calc_balanced_accuracy(calc_signals, sims_mean, verbose=False)
        return res

    def print_score(self, score, is_expert):
        # significance test
        pval_binom = 'notImplemented'  # TODO implement
        chance = 0.5
        print('{} {}={:.2f} (chance={:.2f}, p={})'.format(
            'Expert' if is_expert else 'Novice', self.metric, score, chance, pval_binom))

    def to_eval_sims_mat(self, sims_mat):
        return sims_mat

    # //////////////////////////////////////////////////// Overwritten Methods END

    def load_probes(self):
        data_dir = '{}/{}'.format(self.data_name1, self.data_name2) if self.data_name2 is not None else self.data_name1
        p = config.Dirs.tasks / data_dir / '{}_{}.txt'.format(
            config.Corpus.name, config.Corpus.num_vocab)
        probes = []
        probe_relata = []
        with p.open('r') as f:
            for line in f.read().splitlines():
                spl = line.split()
                probes.append(spl[0])
                probe_relata.append(spl[1:])
        return probes, probe_relata

    # ///////////////////////////////////////////////////// figs

    def make_trial_figs(self, trial):
        return make_matching_figs()