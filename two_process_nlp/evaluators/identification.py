import numpy as np
from scipy.stats import binom

from two_process_nlp import config
from two_process_nlp.evaluators.base import EvalBase
from two_process_nlp.scores import calc_accuracy


class IdentificationParams:  # to retrieve a property during runtime: trial.params.property1
    pass


class Identification(EvalBase):
    def __init__(self, arch, data_name1, data_name2, suffix=''):
        super().__init__(arch.name,
                         arch.Params,
                         arch.init_results_data,
                         arch.split_and_vectorize_eval_data,
                         arch.make_graph,
                         arch.train_expert_on_train_fold,
                         arch.train_expert_on_test_fold,
                         'identification', data_name1, data_name2, suffix,
                         IdentificationParams)
        #
        self.num_epochs = config.Eval.num_epochs_identification
        self.num_epochs_in_eval_step = self.num_epochs // config.Eval.num_evals
        #
        self.probe2relata = None
        self.probe2lures = None
        self.metric = 'acc'
        self.suffix = suffix
        if suffix != '':
            print('WARNING: Using task file suffix "{}".'.format(suffix))

    # ///////////////////////////////////////////// Overwritten Methods START

    def make_all_eval_data(self, vocab_sims_mat, vocab, verbose=False):
        """
        actual evaluation data is sampled from result of this method
        """

        def sample_candidates(name, population, num):
            np.random.seed(43)
            try:
                return np.random.choice(population, num, replace=False).tolist()
            except ValueError:
                if verbose:
                    print('Skipping "{}". Not enough {} (needed={}, available={})'.format(
                        probe, name, num, len(population)))
                return []
        # load
        probes, probe_relata, probe_lures = self.load_probes()
        self.probe2relata = {p: r for p, r in zip(probes, probe_relata)}
        self.probe2lures = {p: l for p, l in zip(probes, probe_lures)}
        # make candidates_mat
        all_eval_probes = []
        all_eval_candidates_mat = []
        num_skipped = 0
        for n, probe in enumerate(probes):
            relata = sample_candidates('relata', self.probe2relata[probe], config.Eval.num_relata)
            lures = sample_candidates('lures', self.probe2lures[probe], config.Eval.num_lures)
            if not relata or not lures:
                num_skipped += 1
                if verbose:
                    print()
                continue
            candidates = relata + lures  # relata must be first (because correct answers are assumed first)
            all_eval_probes.append(probe)
            all_eval_candidates_mat.append(candidates)
            #
            if verbose:
                print(probe)
                print(candidates)
                print()
        if config.Eval.verbose:
            print('Skipped {} probes due to insufficient relata or lures'.format(num_skipped))
        all_eval_candidates_mat = np.vstack(all_eval_candidates_mat)
        return all_eval_probes, all_eval_candidates_mat

    def check_negative_example(self, trial, p=None, c=None):
        assert p is not None
        assert c is not None
        #
        if c in self.probe2lures[p]:
            return True
        else:
            raise RuntimeError('Example must be a lure but is not.')

    def score(self, eval_sims_mat):
        res = calc_accuracy(eval_sims_mat, self.row_words, self.eval_candidates_mat)
        return res

    def print_score(self, score, num_epochs=None):
        # significance test
        # 1-tailed: is observed proportion higher than chance?
        # pval=1.0 when prop=0 because it is almost always true that a prop > 0.0 is observed
        # assumes that proportion of correct responses is binomially distributed
        chance = 1 / 2  # because each multi-answer question is broken down into a 2-part question
        n = len(self.row_words)
        pval_binom = 1 - binom.cdf(k=score * n, n=n, p=chance)
        # console
        print('{} {}={:.2f} (chance={:.2f}, p={}) {}'.format(
            'Expert' if num_epochs is not None else 'Novice',
            self.metric, score, chance, pval_binom,
            'at epoch={}'.format(num_epochs) if num_epochs is not None else ''))

    def to_eval_sims_mat(self, sims_mat):
        res = np.full_like(self.eval_candidates_mat, np.nan, dtype=float)
        for i, candidates_row in enumerate(self.eval_candidates_mat):
            eval_probe = self.row_words[i]
            for j, candidate in enumerate(candidates_row):
                row_id = self.row_words.index(eval_probe)  # can use .index() because duplicate sim rows are identical
                col_id = self.col_words.index(candidate)
                res[i, j] = sims_mat[row_id, col_id]
        return res

    # ///////////////////////////////////////// Overwritten Methods END

    def load_probes(self):
        # get paths to relata and lures
        p1 = config.LocalDirs.tasks / self.data_name1 / self.data_name2 / '{}_{}{}.txt'.format(
            config.Corpus.name, config.Corpus.num_vocab, self.suffix)
        p2s = [p for p in (config.LocalDirs.tasks / self.data_name1).rglob('{}_{}{}.txt'.format(
            config.Corpus.name, config.Corpus.num_vocab, self.suffix)) if p != p1]
        if len(p2s) != 1:
            print(p2s)
            raise ValueError('Found more or less than 1 item text files which is supposed to contain lures.')
        else:
            p2 = p2s[0]
        # load files
        probes2relata1 = {}
        with p1.open('r') as f:
            for line in f.read().splitlines():
                spl = line.split()
                probes2relata1[spl[0]] = spl[1:]
        probes2relata2 = {}
        with p2.open('r') as f:
            for line in f.read().splitlines():
                spl = line.split()
                probes2relata2[spl[0]] = spl[1:]
        # get probes for which lures exist
        probes = []
        probe_relata = []
        probe_lures = []
        for probe, relata in sorted(probes2relata1.items(), key=lambda i: i[0]):
            if probe in probes2relata2:
                probes.append(probe)
                probe_relata.append(relata)
                probe_lures.append(probes2relata2[probe])
            else:
                print('"{}" does not occur in both task files.'.format(probe))
        # no need for downsampling because identification eval is much less memory consuming
        # (identification evaluation is for testing hypotheses about training on specifically selected negative pairs)
        # rather than on all possible negative pairs - this provides a strong test of the idea that
        # it matter more WHICH negative pairings are trained, rather than HOW MANY)
        # no need for shuffling because shuffling is done when eval data is created
        print('{}: After finding lures, number of probes left={}'.format(self.name, len(probes)))
        return probes, probe_relata, probe_lures