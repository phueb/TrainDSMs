import numpy as np
import tensorflow as tf
from scipy.stats import binom
import time
import pandas as pd

from src import config
from src.figs import make_identification_figs
from src.evals.base import EvalBase


class Params:
    shuffled = [False, True]
    margin = [50.0, 100.0]  # must be float and MUST be at least 40 or so
    train_on_second_neighbors = [True]  # performance is much better with additional training
    beta = [0.1]
    num_output = [30]  # TODO
    mb_size = [4]  # TODO what's a good value?
    num_epochs = [500]
    learning_rate = [0.1]


class Identification(EvalBase):
    def __init__(self, data_name1, data_name2):  # data_name2 is not optional because lures must be specified
        name = '{}_{}_identification'.format(data_name1, data_name2)  # reversed is okay
        super().__init__(name, Params)
        #
        self.data_name1 = data_name1
        self.data_name2 = data_name2
        #
        self.probes, self.probe_bros, self.probe_lures = self.load_probes()
        self.probe2bros = {p: bro for p, bro in zip(self.probes, self.probe_bros)}
        self.bros = np.unique(np.concatenate(self.probe_bros)).tolist()
        self.lures = np.unique(np.concatenate(self.probe_lures)).tolist()
        self.num_probes = len(self.probes)
        self.num_bros = len(self.bros)
        self.num_lures = len(self.lures)
        # sims
        self.row_words = self.probes
        self.col_words = sorted(set(self.probes + self.bros + self.lures))
        # evaluation
        self.eval_probes = None
        self.eval_candidates_mat = None

    # ///////////////////////////////////////////// Overwritten Methods START

    def init_eval_data(self, trial):
        res = super().init_eval_data(trial)
        res.test_acc_trajs = np.zeros((config.Eval.num_evals, config.Eval.num_folds))
        return res

    def make_data(self, trial, w2e, fold_id):
        # train/test split
        x1_train = []
        x2_train = []
        y_train = []
        x1_test = []
        x2_test = []
        for n, (test_probes, candidate_rows) in enumerate(zip(
                np.array_split(self.eval_probes, config.Eval.num_folds),
                np.array_split(self.eval_candidates_mat, config.Eval.num_folds))):
            bros, lures, first_neighbors, second_neighbors, neutrals = candidate_rows.T
            if n != fold_id:
                x1_train += [w2e[p] for p in test_probes] + [w2e[p] for p in test_probes]
                x2_train += [w2e[n] for n in bros] + [w2e[d] for d in lures]
                y_train += [1] * len(test_probes) + [0] * len(test_probes)
                if trial.params.train_on_second_neighbors:
                    x1_train += [w2e[p] for p in test_probes] + [w2e[p] for p in test_probes]
                    x2_train += [w2e[n] for n in bros] + [w2e[d] for d in second_neighbors]
                    y_train += [1] * len(test_probes) + [0] * len(test_probes)
            else:
                # assume first one is always correct
                dim1 = candidate_rows.shape[1]
                x1_test += [[w2e[p]] * dim1 for p in test_probes]
                x2_test += [[w2e[bro], w2e[lure], w2e[fn], w2e[sn], w2e[n]] for bro, lure, fn, sn, n in zip(
                        bros, lures, first_neighbors, second_neighbors, neutrals)]
        x1_train = np.vstack(x1_train)
        x2_train = np.vstack(x2_train)
        y_train = np.array(y_train)
        x1_test = np.array(x1_test)
        x2_test = np.array(x2_test)
        # shuffle x-y mapping
        if trial.params.shuffled:
            print('Shuffling supervisory signal')
            np.random.shuffle(y_train)
        return x1_train, x2_train, y_train, x1_test, x2_test

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
        x1_train, x2_train, y_train, x1_test, x2_test = data
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
                # train loss - can't evaluate accuracy because training requires 1:1 match vs. non-match
                train_loss = graph.sess.run(graph.loss, feed_dict={graph.x1: x1_train,
                                                                   graph.x2: x2_train,
                                                                   graph.y: y_train})

                # test acc - use multiple x2 (unlike training)
                num_correct_test = 0
                for x1_mat, x2_mat in zip(x1_test, x2_test):
                    eucd = graph.sess.run(graph.eucd, feed_dict={graph.x1: x1_mat,
                                                                 graph.x2: x2_mat})
                    if np.argmin(eucd) == 0:  # first one is always correct
                        num_correct_test += 1
                test_acc = num_correct_test / num_test_probes
                trial.eval.test_acc_trajs[eval_id, fold_id] = test_acc
                print('step {:>6,}/{:>6,} |Train Loss={:>7.2f} |Test Acc={:.2f} p={:.4f} secs={:.1f}'.format(
                    step,
                    num_train_steps - 1,
                    train_loss, test_acc,
                    self.pval(test_acc, n=num_test_probes),
                    time.time() - start))
                start = time.time()
            # train
            graph.sess.run([graph.step], feed_dict={graph.x1: x1_batch, graph.x2: x2_batch, graph.y: y_batch})

    def get_best_trial_score(self, trial):
        mean_test_acc_traj = trial.eval.test_acc_trajs.mean(axis=1)
        best_eval_id = np.argmax(mean_test_acc_traj)
        expert_score = mean_test_acc_traj[best_eval_id]
        print('Expert score={:.2f} (at eval step {})'.format(expert_score, best_eval_id))
        return expert_score

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
        # get paths to bros and lures
        p1 = config.Dirs.tasks / self.data_name1 / self.data_name2 / '{}_{}.txt'.format(
            config.Corpus.name, config.Corpus.num_vocab)
        p2s = [p for p in (config.Dirs.tasks / self.data_name1).rglob('{}_{}.txt'.format(
            config.Corpus.name, config.Corpus.num_vocab)) if p != p1]
        if len(p2s) != 1:
            raise ValueError('Found more or less than 1 item text files which is supposed to contain lures.')
        else:
            p2 = p2s[0]
        # load files
        probes2bros1 = {}
        with p1.open('r') as f:
            for line in f.read().splitlines():
                spl = line.split()
                probes2bros1[spl[0]] = spl[1:]
        probes2bros2 = {}
        with p2.open('r') as f:
            for line in f.read().splitlines():
                spl = line.split()
                probes2bros2[spl[0]] = spl[1:]
        # get probes for which lures exist
        zipped = []
        for probe, bros in probes2bros1.items():
            if probe in probes2bros2:
                zipped.append([probe, bros, probes2bros2[probe]])
        # shuffle
        np.random.shuffle(zipped)
        # (no need for downsampling because identification eval is much less memory consuming)
        # (identification eval is for testing hypotheses about training on specifically selected negative pairs)
        # (rather than on all possible negative pairs - this provides a strong test of the idea that
        # it matter more WHICH negative pairings are trained, rather than HOW MANY
        probes, probe_brothers, probe_lures = zip(*zipped)
        return list(probes), list(probe_brothers), list(probe_lures)

    def make_eval_candidates_mat(self, sims, verbose=True):
        # make lures and bros
        if config.Eval.remove_duplicates_for_identification:
            eval_probes = self.probes
            eval_bros = [np.random.choice(bros, 1)[0] for bros in self.probe_bros]
            eval_lures = [np.random.choice(ls, 1)[0] for ls in self.probe_lures]
        else:
            eval_probes = []
            eval_bros = []
            eval_lures = []
            for probe, probe_bros, probe_lures in zip(self.probes, self.probe_bros, self.probe_lures):
                for bro, lure in zip(probe_bros, probe_lures):
                    eval_probes.append(probe)
                    eval_bros.append(bro)
                    eval_lures.append(lure)
        # get neutrals + neighbors
        neutrals = np.roll(eval_bros, 1)
        first_neighbors = []
        second_neighbors =[]
        for p in eval_probes:  # otherwise argmax would return index for the probe itself
            row_id = self.probes.index(p)
            ids = np.argsort(sims[row_id])[::-1][:10]
            nearest_neighbors = [self.col_words[i] for i in ids]
            first_neighbors.append(nearest_neighbors[1])  # don't use id=zero because it returns the probe
            second_neighbors.append(nearest_neighbors[2])
        # make mat
        eval_candidates_mat = []
        num_eval_probes = len(eval_probes)
        for i in range(num_eval_probes):
            if first_neighbors[i] == eval_bros[i]:
                first_neighbors[i] = neutrals[i]
            if second_neighbors[i] == eval_bros[i]:
                second_neighbors[i] = neutrals[i]
            candidates = [eval_bros[i], eval_lures[i], first_neighbors[i], second_neighbors[i], neutrals[i]]
            if verbose:
                print(eval_probes[i])
                print(candidates)
            eval_candidates_mat.append(candidates)
        eval_candidates_mat = np.vstack(eval_candidates_mat)
        return eval_probes, eval_candidates_mat

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

    def score_novice(self, sims, verbose=False):
        self.eval_probes, self.eval_candidates_mat = self.make_eval_candidates_mat(sims)  # constructed here because it requires sims
        num_correct = 0
        for eval_probe, eval_candidates in zip(self.eval_probes, self.eval_candidates_mat):
            row_id = self.probes.index(eval_probe)
            candidate_sims = sims[row_id, [self.col_words.index(w) for w in eval_candidates]]
            if verbose:
                print(np.around(candidate_sims, 2))
            if np.all(candidate_sims[1:] < candidate_sims[0]):  # correct is always in first position
                num_correct += 1
        self.novice_score = num_correct / self.num_probes
        print('Novice Accuracy={:.2f} (chance={:.2f}, p={:.4f})'.format(
            self.novice_score,
            self.chance,
            self.pval(self.novice_score, n=self.num_probes)))
