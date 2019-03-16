import numpy as np
from scipy.stats import binom
import pickle

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
        self.probe2cats = None
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
        probes, probe_relata, probe_lures, probe_cats = self.load_probes()
        self.probe2relata = {p: r for p, r in zip(probes, probe_relata)}
        self.probe2lures = {p: l for p, l in zip(probes, probe_lures)}
        self.probe2cats = {p: cats for p, cats in zip(probes, probe_cats)}
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

    def save_task_meta_data(self, row_words, embedder_location, process):
        assert self.probe2relata is not None
        assert self.probe2lures is not None
        #
        metadata = []  # must be ordered like row_words
        for rw in row_words + config.Eval.tertiary_probes:
            try:
                cats = self.probe2cats[rw]
            except KeyError:  # tertiary probe
                cats = ['TERTIARY']
            metadata.append((rw, cats))
        #
        p = self.make_p(embedder_location, process, 'task_metadata.pkl')
        with p.open('wb') as f:
            pickle.dump(metadata, f)

    def check_negative_example(self, trial, p=None, c=None):
        assert p is not None
        assert c is not None
        #
        if c in self.probe2lures[p]:
            if trial.params.neg_pos_ratio == 1.0:
                return True
            elif trial.params.neg_pos_ratio == 0.0:
                return False
            else:
                raise AttributeError('"neg_pos_ratio" for identification must be either 0 or 1.')
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
        # load data

        def opposite(cat_loading):
            if cat_loading == '+':
                return '-'
            elif cat_loading == '-':
                return '+'
            else:
                raise AttributeError('Invalid arg to "cat_loading".')

        def populate_dicts(f_handle, d1, d2):
            for line in f_handle.read().splitlines():
                spl = line.split()
                probe = spl[0]
                relata = spl[1:-1]
                cat = spl[-1]  # category is in last position
                cat_name, cat_loading = cat[:-1], cat[-1]
                # relata
                d1[probe] = relata
                # cat
                for w in [probe] + relata:  # the same word can be a probe or relatum (and therefore have > 1 category)
                    if w in d2:
                        if not (cat_name, opposite(cat_loading)) in d2[w]:  # TODO test
                            d2[w].add((cat_name, cat_loading))
                    else:
                        d2[w] = set()
                        d2[w].add((cat_name, cat_loading))
            return d1, d2

        probe2relata1 = {}
        probe2cats = {}  # a probe can have multiple categories  # TODO test
        with p1.open('r') as f:
            probe2relata1, probe2cats = populate_dicts(f, probe2relata1, probe2cats)
        probe2relata2 = {}
        with p2.open('r') as f:
            probe2relata2, probe2cats = populate_dicts(f, probe2relata2, probe2cats)
        # get probes for which lures exist
        probes = []
        probe_relata = []
        probe_lures = []
        probe_cats = []
        for probe, relata in sorted(probe2relata1.items(), key=lambda i: i[0]):
            if probe in probe2relata2:
                probes.append(probe)
                probe_relata.append(relata)
                probe_lures.append(probe2relata2[probe])
                probe_cats.append([cat_name + cat_loading for (cat_name, cat_loading) in probe2cats[probe]])
            else:
                print('"{}" does not occur in both task files.'.format(probe))
        # no need for downsampling because identification eval is much less memory consuming
        # (identification evaluation is for testing hypotheses about training on specifically selected negative pairs)
        # rather than on all possible negative pairs - this provides a strong test of the idea that
        # it matter more WHICH negative pairings are trained, rather than HOW MANY)
        # no need for shuffling because shuffling is done when eval data is created
        print('{}: After finding lures, number of probes left={}'.format(self.name, len(probes)))
        return probes, probe_relata, probe_lures, probe_cats