import numpy as np
import tensorflow as tf
from scipy.stats import binom
import time
from itertools import product

from src import config
from src.figs import make_identification_figs
from src.evals.base import EvalBase
from src.scores import calc_accuracy


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
        self.row_words = probes
        self.col_words = sorted(set(probes + relata + lures))  # must be set even if each is a set

    # ///////////////////////////////////////////// Overwritten Methods START

    def init_results_data(self, trial):
        """
        add task-specific attributes to EvalData class implemented in TaskBase
        """
        res = super().init_results_data(trial)
        res.eval_sims_mats = [np.zeros_like(self.eval_candidates_mat, float)  # same shape as eval_candidates_mat TODO test
                              for _ in range(config.Eval.num_evals)]
        return res

    def make_eval_data(self, sims, verbose=False):  # TODO inefficient: eval_probes are always row_words
        # make lures and relata
        if config.Eval.remove_duplicates_for_identification:
            eval_relata = [np.random.choice(self.probe2relata[p], 1)[0] for p in  self.row_words]
            eval_lures = [np.random.choice(self.probe2lures[p], 1)[0] for p in  self.row_words]
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

    def check_negative_example(self, trial, p, c):
        if c in self.probe2lures[p]:
            return True
        elif c in self.eval_candidates_mat[:, 3]:  # relata, lures, fns, sns, neutrals
            if trial.params.train_on_second_neighbors:
                return True
            else:
                return False
        else:
            return False

    def split_and_vectorize_eval_data(self, trial, w2e, fold_id):  # TODO collapse with matching eval function
        # split
        x1_train = []
        x2_train = []
        y_train = []
        x1_test = []
        x2_test = []
        test_probes = []
        for n, (eval_probes, candidate_rows) in enumerate(zip(
                np.array_split(self.row_words, config.Eval.num_folds),
                np.array_split(self.eval_candidates_mat, config.Eval.num_folds))):
            if n != fold_id:
                for probe, candidates in zip(eval_probes, candidate_rows):
                    for p, c in product([probe], candidates):
                        if c in self.probe2relata[p] or self.check_negative_example(trial, p, c):
                            x1_train.append(w2e[probe])
                            x2_train.append(w2e[c])
                            y_train.append(1 if c in self.probe2relata[p] else 0)
            else:
                # test data to build chunk of eval_sim_mat
                for probe, candidates in zip(eval_probes, candidate_rows):
                    x1_test += [[w2e[probe]] * len(candidates)]  # TODO test
                    x2_test += [[w2e[c] for c in candidates]]
                    test_probes.append(probe)
        x1_train = np.vstack(x1_train)
        x2_train = np.vstack(x2_train)
        y_train = np.array(y_train)
        x1_test = np.array(x1_test)
        x2_test = np.array(x2_test)
        # shuffle x-y mapping
        if trial.params.shuffled:
            print('Shuffling supervisory signal')
            np.random.shuffle(y_train)
        return x1_train, x2_train, y_train, x1_test, x2_test, test_probes

    def make_graph(self, trial, embed_size):

        def siamese_leg(x, wy):
            y = tf.matmul(x, wy)
            return y

        class Graph:
            with tf.Graph().as_default():
                with tf.device('/{}:0'.format(config.Eval.device)):
                    # placeholders
                    x1 = tf.placeholder(tf.float32, shape=(None, embed_size))
                    x2 = tf.placeholder(tf.float32, shape=(None, embed_size))
                    y = tf.placeholder(tf.float32, [None])
                    # siamese
                    with tf.variable_scope('trial_{}'.format(trial.params_id), reuse=tf.AUTO_REUSE) as scope:
                        wy = tf.get_variable('wy', shape=[embed_size, trial.params.num_output], dtype=tf.float32)
                        o1 = siamese_leg(x1, wy)
                        o2 = siamese_leg(x2, wy)
                    # loss
                    labels_t = y
                    labels_f = tf.subtract(1.0, y)
                    eucd2 = tf.pow(tf.subtract(o1, o2), 2)
                    eucd2 = tf.reduce_sum(eucd2, 1)
                    eucd = tf.sqrt(eucd2 + 1e-6)
                    C = tf.constant(trial.params.margin)
                    # yi*||o1-o2||^2 + (1-yi)*max(0, C-||o1-o2||^2)
                    pos = tf.multiply(labels_t, eucd2)
                    neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2))
                    losses = tf.add(pos, neg)
                    loss_no_reg = tf.reduce_mean(losses)
                    regularizer = tf.nn.l2_loss(wy)
                    loss = tf.reduce_mean((1 - trial.params.beta) * loss_no_reg +
                                          trial.params.beta * regularizer)
                    optimizer = tf.train.AdadeltaOptimizer(learning_rate=trial.params.learning_rate)
                    step = optimizer.minimize(loss)
                # session
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
        return Graph()

    def train_expert_on_train_fold(self, trial, graph, data, fold_id):
        x1_train, x2_train, y_train, x1_test, x2_test, test_probes = data
        num_train_probes, num_test_probes = len(x1_train), len(x1_test)
        num_train_steps = num_train_probes // trial.params.mb_size * trial.params.num_epochs
        eval_interval = num_train_steps // config.Eval.num_evals
        eval_steps = np.arange(0, num_train_steps + eval_interval,
                               eval_interval)[:config.Eval.num_evals].tolist()  # equal sized intervals
        print('Train data size: {:,} | Test data size: {:,}'.format(num_train_probes, num_test_probes))
        # training and eval
        start = time.time()
        for step, x1_batch, x2_batch, y_batch in self.generate_random_train_batches(x1_train,
                                                                                    x2_train,
                                                                                    y_train,
                                                                                    num_train_probes,
                                                                                    num_train_steps,
                                                                                    trial.params.mb_size):
            if step in eval_steps:
                eval_id = eval_steps.index(step)
                # train loss
                train_loss = graph.sess.run(graph.loss, feed_dict={graph.x1: x1_train,
                                                                   graph.x2: x2_train,
                                                                   graph.y: y_train})

                # test acc cannot be computed because only partial probe sims mat is available
                for x1_mat, x2_mat, test_probe in zip(x1_test, x2_test, test_probes):
                    eucd = graph.sess.run(graph.eucd, feed_dict={graph.x1: x1_mat,
                                                                 graph.x2: x2_mat})
                    eval_sims_mat_row = 1.0 - (eucd / trial.params.margin)
                    row_id = self.row_words.index(test_probe)
                    trial.results.eval_sims_mats[eval_id][row_id, :] = eval_sims_mat_row  # TODO test
                print('step {:>6,}/{:>6,} |Train Loss={:>7.2f} |secs={:.1f}'.format(
                    step,
                    num_train_steps - 1,
                    train_loss,
                    time.time() - start))
                start = time.time()
            # train
            graph.sess.run([graph.step], feed_dict={graph.x1: x1_batch, graph.x2: x2_batch, graph.y: y_batch})

    def train_expert_on_test_fold(self, trial, graph, data, fold_id):
        raise NotImplementedError

    def get_best_trial_score(self, trial):
        best_expert_score = 0
        best_eval_id = 0
        for eval_id, eval_sims_mat in enumerate(trial.results.eval_sims_mats):
            expert_score = calc_accuracy(eval_sims_mat, self.row_words, self.eval_candidates_mat)  # TODO test fn
            print('acc at eval {} is {:.2f}'.format(eval_id + 1, expert_score))
            if expert_score > best_expert_score:
                best_expert_score = expert_score
                best_eval_id = eval_id
        print('Expert score={:.2f} (at eval step {})'.format(best_expert_score, best_eval_id + 1))
        return best_expert_score

    def make_trial_figs(self, trial):
        return make_identification_figs()

    # ///////////////////////////////////////// Overwritten Methods END

    @staticmethod
    def generate_random_train_batches(x1, x2, y, num_probes, num_steps, mb_size):
        random_choices = np.random.choice(num_probes, mb_size * num_steps)
        row_ids_list = np.split(random_choices, num_steps)
        for n, row_ids in enumerate(row_ids_list):
            assert len(row_ids) == mb_size
            x1_batch = x1[row_ids]
            x2_batch = x2[row_ids]
            y_batch = y[row_ids]
            yield n, x1_batch, x2_batch, y_batch

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

    @property
    def chance(self):
        res = 1 / self.eval_candidates_mat.shape[1]
        return res

    def pval(self, prop, n):
        """
        1-tailed: is observed proportion higher than chance?
        pval=1.0 when prop=0 because it is almost always true that a prop > 0.0 is observed
        assumes that proportion of correct responses is binomially distributed
        """
        pval_binom = 1 - binom.cdf(k=prop * n, n=n, p=self.chance)
        return pval_binom

    def score_novice(self, eval_sims_mat):
        # score
        self.novice_score = calc_accuracy(eval_sims_mat, self.row_words, self.eval_candidates_mat)
        print('Novice Accuracy={:.2f} (chance={:.2f}, p={:.4f})'.format(
            self.novice_score,
            self.chance,
            self.pval(self.novice_score, n=len(self.row_words))))
