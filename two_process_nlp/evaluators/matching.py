import numpy as np
from functools import partial

from two_process_nlp import config
from two_process_nlp.scores import calc_cluster_score
from two_process_nlp.evaluators.base import EvalBase


class MatchingParams:
    pass


class Matching(EvalBase):
    def __init__(self, arch, data_name1, data_name2='', suffix=''):
        super().__init__(arch.name,
                         arch.Params,
                         arch.init_results_data,
                         arch.split_and_vectorize_eval_data,
                         arch.make_graph,
                         arch.train_expert_on_train_fold,
                         arch.train_expert_on_test_fold,
                         'matching', data_name1, data_name2, suffix,
                         MatchingParams)
        #
        self.binomial = np.random.binomial
        self.metric = config.Eval.matching_metric
        self.suffix = suffix

    # ///////////////////////////////////////////// Overwritten Methods START

    def make_all_eval_data(self, vocab_sims_mat, vocab):  # vocab_sims_mat is used in identification eval
        # load
        probes, probe_relata = self.load_probes()  # relata can be synonyms, hypernyms, etc.
        relata = sorted(np.unique(np.concatenate(probe_relata)).tolist())
        self.probe2relata = {p: r for p, r in zip(probes, probe_relata)}
        #
        all_eval_probes = probes
        num_eval_probes = len(all_eval_probes)
        all_eval_candidates_mat = np.asarray([relata for _ in range(num_eval_probes)])
        return all_eval_probes, all_eval_candidates_mat

    def check_negative_example(self, trial, p=None, c=None):
        if c in self.probe2relata[p]:
            raise RuntimeError('While checking negative example, found positive example')
        if trial.params.neg_pos_ratio == 1.0:  # balance positive : negative  approximately  1 : 1
            prob = self.pos_prob
        else:
            neg_prob = self.pos_prob * trial.params.neg_pos_ratio
            if neg_prob > 0.5:
                raise ValueError('Setting "neg_pos_ratio" would result in negative_prob > 0.5 ({}).'.format(neg_prob))
            prob = min(1.0, neg_prob)
        return bool(self.binomial(n=1, p=prob, size=1))

    def score(self, eval_sims_mat, verbose=False):
        # the random-control embedder sim mean should be higher because it can learn global properties of data only
        if verbose:
            print('Mean of eval_sims_mat={:.4f}'.format(eval_sims_mat.mean()))
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
        res = calc_cluster_score(calc_signals, sims_mean, verbose=False)
        return res

    def print_score(self, score, num_epochs=None):
        # significance test
        pval_binom = 'notImplemented'  # TODO implement
        chance = 0.5
        print('{} {}={:.2f} (chance={:.2f}, p={}) {}'.format(
            'Expert' if num_epochs is not None else 'Novice',
            self.metric, score, chance, pval_binom,
            'at epoch={}'.format(num_epochs) if num_epochs is not None else ''))

    def to_eval_sims_mat(self, sims_mat):
        return sims_mat

    # //////////////////////////////////////////////////// Overwritten Methods END

    def load_probes(self):
        data_dir = '{}/{}'.format(self.data_name1, self.data_name2) if self.data_name2 is not None else self.data_name1
        p = config.LocalDirs.tasks / data_dir / '{}_{}{}.txt'.format(
            config.Corpus.name, config.Corpus.num_vocab, self.suffix)
        probes1 = []
        probe_relata1 = []
        num_relata_list = []
        # read file
        with p.open('r') as f:
            for line in f.read().splitlines():
                spl = line.split()
                probe = spl[0]
                relata = spl[1:]
                probes1.append(probe)
                probe_relata1.append(relata)
                num_relata_list.append(len(relata))
        # keep number of relata constant  - this is not ideal due to small size of data (relata)
        min_num_relata = min(num_relata_list)
        if config.Eval.standardize_num_relata:
            print('WARNING: Standardizing number of relata.')
            probes2 = []
            probe_relata2 = []
            for probe, relata in zip(probes1, probe_relata1):
                probes2.append(probe)
                probe_relata2.append(relata[:min_num_relata])
        else:
            probes2 = probes1
            probe_relata2 = probe_relata1
        return probes2, probe_relata2