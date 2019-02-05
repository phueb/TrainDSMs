import numpy as np
from scipy.stats import binom

from two_stage_nlp import config
from two_stage_nlp.evaluators.base import EvalBase
from two_stage_nlp.scores import calc_accuracy


class IdentificationParams:
    only_positive_examples = [False, True]  # performance can be much better without negative examples
    train_on_second_neighbors = [True]  # performance can be much better with additional training
    # arch-evaluator interaction
    num_epochs = [2000]  # 2000 is better than 100 or 200, 300, 500, 1000  but not 5000


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
        self.probe2relata = None
        self.probe2lures = None
        self.probe2sns = None
        self.metric = 'acc'
        self.suffix = suffix
        if suffix != '':
            print('WARNING: Using task file suffix "{}".'.format(suffix))

    # ///////////////////////////////////////////// Overwritten Methods START

    def make_all_eval_data(self, vocab_sims_mat, vocab, verbose=False):
        """
        actual evaluation data is sampled from result of this method
        """
        # load
        probes, probe_relata, probe_lures = self.load_probes()
        self.probe2relata = {p: r for p, r in zip(probes, probe_relata)}
        self.probe2lures = {p: l for p, l in zip(probes, probe_lures)}
        self.probe2sns = {p: set() for p in probes}
        # make lures and relata
        if config.Eval.remove_duplicates_for_identification:
            all_eval_probes = probes  # leave this - eval_probes is different from probes if duplicates are not removed
            eval_relata = [np.random.choice(self.probe2relata[p], 1)[0] for p in all_eval_probes]
            eval_lures = [np.random.choice(self.probe2lures[p], 1)[0] for p in all_eval_probes]
        else:
            all_eval_probes = []
            eval_relata = []
            eval_lures = []
            for probe in probes:
                for relatum, lure in zip(self.probe2relata[probe], self.probe2lures[probe]):
                    all_eval_probes.append(probe)
                    eval_relata.append(relatum)
                    eval_lures.append(lure)
        # get neutrals + neighbors
        num_roll = len(eval_relata) // 2
        assert num_roll > max([len(i) for i in self.probe2relata.values()])
        neutrals = np.roll(eval_relata, num_roll)  # avoids relatum to be wrong answer in next pair
        first_neighbors = []
        second_neighbors =[]
        for p in all_eval_probes:
            row_id = vocab.index(p)
            ids = np.argsort(vocab_sims_mat[row_id])[::-1][:10]
            nearest_neighbors = [vocab[i] for i in ids]
            first_neighbors.append(nearest_neighbors[1])  # don't use id=zero because it returns the probe
            second_neighbors.append(nearest_neighbors[2])
        # make candidates_mat
        all_eval_candidates_mat = []
        for n, probe in enumerate(all_eval_probes):
            if first_neighbors[n] == eval_relata[n]:  # TODO doesn't exclude alternative spellings or morphologies (or alternative synonyms)
                first_neighbors[n] = neutrals[n]
            if second_neighbors[n] == eval_relata[n]:
                second_neighbors[n] = neutrals[n]
            candidates = [eval_relata[n], eval_lures[n], first_neighbors[n], second_neighbors[n], neutrals[n]]
            self.probe2sns[probe].add(second_neighbors[n])
            if verbose:
                print(all_eval_probes[n])
                print(candidates)
            all_eval_candidates_mat.append(candidates)
        all_eval_candidates_mat = np.vstack(all_eval_candidates_mat)
        return all_eval_probes, all_eval_candidates_mat

    def check_negative_example(self, trial, p=None, c=None):
        assert p is not None
        assert c is not None
        #
        if trial.params.only_positive_examples:
            return False  # this is much better than training on negatives
        #
        if c in self.probe2lures[p]:
            return True
        elif c in self.probe2sns[p]:
            if trial.params.train_on_second_neighbors:
                return True
            else:
                return False
        else:
            return False

    def score(self, eval_sims_mat):
        res = calc_accuracy(eval_sims_mat, self.row_words, self.eval_candidates_mat)
        return res

    def print_score(self, score, eval_id=None):
        # significance test
        # 1-tailed: is observed proportion higher than chance?
        # pval=1.0 when prop=0 because it is almost always true that a prop > 0.0 is observed
        # assumes that proportion of correct responses is binomially distributed
        chance = 1 / self.eval_candidates_mat.shape[1]
        n = len(self.row_words)
        pval_binom = 1 - binom.cdf(k=score * n, n=n, p=chance)
        # console
        print('{} {}={:.2f} (chance={:.2f}, p={}) {}'.format(
            'Expert' if eval_id is not None else 'Novice',
            self.metric, score, chance, pval_binom,
            'at eval={}'.format(eval_id + 1) if eval_id is not None else ''))

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
        p1 = config.Dirs.tasks / self.data_name1 / self.data_name2 / '{}_{}{}.txt'.format(
            config.Corpus.name, config.Corpus.num_vocab, self.suffix)
        p2s = [p for p in (config.Dirs.tasks / self.data_name1).rglob('{}_{}{}.txt'.format(
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
        for probe, relata in probes2relata1.items():
            if probe in probes2relata2:
                probes.append(probe)
                probe_relata.append(relata)
                probe_lures.append(probes2relata2[probe])
        # no need for downsampling because identification eval is much less memory consuming
        # (identification evaluation is for testing hypotheses about training on specifically selected negative pairs)
        # rather than on all possible negative pairs - this provides a strong test of the idea that
        # it matter more WHICH negative pairings are trained, rather than HOW MANY)
        # no need for shuffling because shuffling is done when eval data is created
        print('{}: After finding lures, number of probes left={}'.format(self.name, len(probes)))  # TODO not many probes leftover
        return probes, probe_relata, probe_lures