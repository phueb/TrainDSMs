import numpy as np
from scipy.stats import binom

from src import config
from src.figs import make_identification_figs
from src.evals.base import EvalBase
from src.scores import calc_accuracy

# TODO make two params - 1 for each arch - switch between the two?
class Params:
    shuffled = [False, True]
    margin = [50.0, 100.0]  # must be float and MUST be at least 40 or so
    train_on_second_neighbors = [True]  # performance is much better with additional training
    beta = [0.0, 0.2]
    num_output = [100]
    mb_size = [4]  # TODO what's a good value?
    num_epochs = [100]
    learning_rate = [0.1]


class Identification(EvalBase):
    def __init__(self, data_name1, data_name2):  # data_name2 is not optional because lures must be specified
        name = '{}_{}_identification'.format(data_name1, data_name2)  # reversed is okay
        super().__init__(name, Params)
        #
        self.data_name1 = data_name1
        self.data_name2 = data_name2
        #
        probes, probe_relata, probe_lures = self.load_probes()
        relata = sorted(np.unique(np.concatenate(probe_relata)).tolist())
        lures = sorted(np.unique(np.concatenate(probe_lures)).tolist())
        self.probe2relata = {p: r for p, r in zip(probes, probe_relata)}
        self.probe2lures = {p: l for p, l in zip(probes, probe_lures)}
        # sims
        self.all_row_words = probes
        self.all_col_words = sorted(set(probes + relata + lures))  # must be set even if each is a set
        # misc
        self.metric = 'acc'

    # ///////////////////////////////////////////// Overwritten Methods START

    def to_eval_sims_mat(self, sims_mat):
        res = np.zeros_like(self.eval_candidates_mat)  # TODO test
        for i, candidates_row in enumerate(self.eval_candidates_mat):
            eval_probe = self.row_words[i]
            for j, candidate in enumerate(candidates_row):
                res[i, j] = sims_mat[self.row_words.index(eval_probe), self.col_words.index(candidate)]
        return res

    def make_eval_data(self, sims, verbose=False):
        # make lures and relata
        if config.Eval.remove_duplicates_for_identification:
            eval_relata = [np.random.choice(self.probe2relata[p], 1)[0] for p in self.row_words]
            eval_lures = [np.random.choice(self.probe2lures[p], 1)[0] for p in self.row_words]
        else:
            eval_relata = []
            eval_lures = []
            for probe in self.row_words:
                for relatum, lure in zip(self.probe2relata[probe], self.probe2lures[probe]):
                    eval_relata.append(relatum)
                    eval_lures.append(lure)
        # get neutrals + neighbors
        num_roll = len(eval_relata) // 2
        assert num_roll > max([len(i) for i in self.probe2relata.values()])
        neutrals = np.roll(eval_relata, num_roll)  # avoids relatum to be wrong answer in next pair
        first_neighbors = []
        second_neighbors =[]
        for p in self.row_words:
            row_id = self.row_words.index(p)
            ids = np.argsort(sims[row_id])[::-1][:10]
            nearest_neighbors = [self.col_words[i] for i in ids]
            first_neighbors.append(nearest_neighbors[1])  # don't use id=zero because it returns the probe
            second_neighbors.append(nearest_neighbors[2])
        # make candidates_mat
        eval_candidates_mat = []
        num_row_words = len(self.row_words)
        for i in range(num_row_words):
            if first_neighbors[i] == eval_relata[i]:
                first_neighbors[i] = neutrals[i]
            if second_neighbors[i] == eval_relata[i]:
                second_neighbors[i] = neutrals[i]
            candidates = [eval_relata[i], eval_lures[i], first_neighbors[i], second_neighbors[i], neutrals[i]]
            if verbose:
                print(self.row_words[i])
                print(candidates)
            eval_candidates_mat.append(candidates)
        eval_candidates_mat = np.vstack(eval_candidates_mat)
        return eval_candidates_mat

    def check_negative_example(self, trial, p=None, c=None):
        assert p is not None
        assert c is not None
        if c in self.probe2lures[p]:
            return True
        elif c in self.eval_candidates_mat[:, 3]:  # relata, lures, fns, sns, neutrals
            if trial.params.train_on_second_neighbors:
                return True
            else:
                return False
        else:
            return False

    def score(self, eval_sims_mat, is_expert):
        # score
        res = calc_accuracy(eval_sims_mat, self.row_words, self.eval_candidates_mat)
        # significance test
        # 1-tailed: is observed proportion higher than chance?
        # pval=1.0 when prop=0 because it is almost always true that a prop > 0.0 is observed
        # assumes that proportion of correct responses is binomially distributed
        chance = 1 / self.eval_candidates_mat.shape[1]
        n = len(self.row_words)
        pval_binom = 1 - binom.cdf(k=res * n, n=n, p=chance)
        # console
        print('{} Accuracy={:.2f} (chance={:.2f}, p={:.4f})'.format(
            'Expert' if is_expert else 'Novice', res, chance, pval_binom))
        return res

    # //////////////////////////////////////////////////// architecture specific

    def init_results_data(self, trial):
        """
        add architecture-specific attributes to EvalData class implemented in TaskBase
        """
        res = super().init_results_data(trial)

        res.arch_name = self.arch.name  # TODO test

        return res

    def split_and_vectorize_eval_data(self, trial, w2e, fold_id):
        raise NotImplementedError  # TODO

    def make_graph(self, trial, embed_size):
        raise NotImplementedError  # TODO

    def train_expert_on_train_fold(self, trial, graph, data, fold_id):
        raise NotImplementedError  # TODO

    def train_expert_on_test_fold(self, trial, graph, data, fold_id):
        raise NotImplementedError

    # ///////////////////////////////////////// Overwritten Methods END

    def load_probes(self):
        # get paths to relata and lures
        p1 = config.Dirs.tasks / self.data_name1 / self.data_name2 / '{}_{}.txt'.format(
            config.Corpus.name, config.Corpus.num_vocab)
        p2s = [p for p in (config.Dirs.tasks / self.data_name1).rglob('{}_{}.txt'.format(
            config.Corpus.name, config.Corpus.num_vocab)) if p != p1]
        if len(p2s) != 1:
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
        # (identification eval is for testing hypotheses about training on specifically selected negative pairs)
        # rather than on all possible negative pairs - this provides a strong test of the idea that
        # it matter more WHICH negative pairings are trained, rather than HOW MANY)
        # no need for shuffling because shuffling is done when eval data is created
        print('{}: After finding lures, number of probes left={}'.format(self.name, len(probes)))  # TODO not many probes leftover
        return probes, probe_relata, probe_lures

    # ////////////////////////////////////////////////////////////// figs

    def make_trial_figs(self, trial):
        return make_identification_figs()