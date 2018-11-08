import numpy as np
import tensorflow as tf
from functools import partial
import pandas as pd

from src import config
from src.utils import calc_balanced_accuracy
from src.figs import make_feature_matching_figs
from src.tasks.base import TaskBase


class Params:
    shuffled = [False, True]
    num_epochs = [500]
    mb_size = [4]
    num_output = [32, 256]
    margin = [100.0]
    beta = [0.0, 0.2]
    learning_rate = [0.1]


NUM_DISTRACTORS = 3


class FeatureMatching(TaskBase):
    def __init__(self, feature_type):
        name = '{}_feature_matching'.format(feature_type)
        super().__init__(name, Params)
        #
        self.feature_type = feature_type
        self.probes, self.probe_features_list = self.load_training_data()
        self.probe2features = {p: fl for p, fl in zip(self.probes, self.probe_features_list)}
        self.features = np.unique(np.concatenate(self.probe_features_list)).tolist()
        self.num_probes = len(self.probes)
        self.num_features = len(self.features)
        # sims
        self.row_words = self.probes
        self.col_words = self.features

    # ///////////////////////////////////////////// Overwritten Methods START

    def init_eval_data(self, trial):
        """
        add task-specific attributes to EvalData class implemented in TaskBase
        """
        res = super().init_eval_data(trial)
        res.test_probe_sims = [np.zeros((self.num_probes, self.num_features))
                               for _ in range(config.Task.num_evals)]
        return res

    def make_data(self, trial, w2e, fold_id):
        # train/test split
        x1_train = []
        x2_train = []
        y_train = []
        x1_test = []
        x2_test = []
        test_probe_ids = []
        probe_distractors_list = [np.random.choice([f for f in self.features if f not in self.probe2features[p]],
                                                   size=NUM_DISTRACTORS, replace=False)
                                  for p in self.probes]
        for n, (probes, features_list, distractors_list) in enumerate(zip(
                np.array_split(self.probes, config.Task.num_folds),
                np.array_split(self.probe_features_list, config.Task.num_folds),
                np.array_split(probe_distractors_list, config.Task.num_folds))):
            if n != fold_id:

                # TODO train on multiple features pre probe not just one

                x1_train += [w2e[p] for p in probes] + [w2e[p] for p in probes]
                x2_train += [w2e[fs[0]] for fs in features_list] + [w2e[ds[0]] for ds in distractors_list]
                y_train += [1] * len(probes) + [0] * len(probes)
            else:
                # test data to build chunk of sim matrix as input to balanced accuracy algorithm
                x1_test += [[w2e[p]] * self.num_features for p in probes]
                x2_test += [[w2e[f] for f in self.features] for _ in probes]
                test_probe_ids += [self.probes.index(p) for p in probes]
        x1_train = np.vstack(x1_train)
        x2_train = np.vstack(x2_train)
        y_train = np.array(y_train)
        x1_test = np.array(x1_test)
        x2_test = np.array(x2_test)
        assert len(x1_train) == len(x2_train) == len(y_train)
        # shuffle x-y mapping
        if trial.params.shuffled:
            print('Shuffling input-output mapping')
            np.random.shuffle(y_train)
        return x1_train, x2_train, y_train, x1_test, x2_test, test_probe_ids

    def make_graph(self, trial, embed_size):

        def siamese_leg(x, wy):
            y = tf.matmul(x, wy)
            return y

        class Graph:
            with tf.Graph().as_default():
                with tf.device('/{}:0'.format(config.Task.device)):
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
        x1_train, x2_train, y_train, x1_test, x2_test, test_probe_ids = data
        num_train_probes, num_test_probes = len(x1_train), len(x1_test)
        num_train_steps = num_train_probes // trial.params.mb_size * trial.params.num_epochs
        eval_interval = num_train_steps // config.Task.num_evals
        eval_steps = np.arange(0, num_train_steps + eval_interval,
                               eval_interval)[:config.Task.num_evals].tolist()  # equal sized intervals
        print('Train data size: {:,} | Test data size: {:,}'.format(num_train_probes, num_test_probes))
        # training and eval
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

                # test acc cannot be computed because only partial probe sims mat is available
                for n, (x1_mat, x2_mat, test_probe_id) in enumerate(zip(x1_test, x2_test, test_probe_ids)):
                    eucd = graph.sess.run(graph.eucd, feed_dict={graph.x1: x1_mat,
                                                                 graph.x2: x2_mat})
                    sims_row = 1.0 - (eucd / trial.params.margin)
                    trial.eval.test_probe_sims[eval_id][test_probe_id, :] = sims_row
                test_bal_acc = 'need to complete all folds'
                print('step {:>6,}/{:>6,} |Train Loss={:>7.2f} |Test BalancedAcc={}'.format(
                    step,
                    num_train_steps - 1,
                    train_loss,
                    test_bal_acc))
            # train
            graph.sess.run([graph.step], feed_dict={graph.x1: x1_batch, graph.x2: x2_batch, graph.y: y_batch})

    def get_best_trial_score(self, trial):
        best_expert_score = 0
        best_eval_id = 0
        for eval_id, sims in enumerate(trial.eval.test_probe_sims):
            calc_signals = partial(self.calc_signals, sims)
            sims_mean = np.asscalar(np.mean(sims))
            expert_score = calc_balanced_accuracy(calc_signals, sims_mean, verbose=False)
            print('{} at eval {} is {:.2f}'.format(config.Task.metric, eval_id + 1, expert_score))
            if expert_score > best_expert_score:
                best_expert_score = expert_score
                best_eval_id = eval_id
        print('Expert score={:.2f} (at eval step {})'.format(best_expert_score, best_eval_id + 1))
        return best_expert_score

    def make_trial_figs(self, trial):
        return make_feature_matching_figs()

    # //////////////////////////////////////////////////// Overwritten Methods END

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

    def load_training_data(self):
        p = config.Dirs.tasks / 'features' / self.feature_type / '{}_{}.txt'.format(
            config.Corpus.name, config.Corpus.num_vocab)
        df = pd.read_csv(p, sep=' ', header=None, error_bad_lines=False)
        df.sample(frac=1)  # shuffle rows
        probes = []
        probe_features = []
        for n, row in df.iterrows():
            probes.append(row[0])
            probe_features.append([f for f in row[1:] if f is not np.nan])
        return probes, probe_features

    def calc_signals(self, probe_sims, thr):
        num_probes = len(probe_sims)
        tp = np.zeros(num_probes, float)
        tn = np.zeros(num_probes, float)
        fp = np.zeros(num_probes, float)
        fn = np.zeros(num_probes, float)
        # calc hits, misses, false alarms, correct rejections
        for i in range(num_probes):
            features1 = self.probe2features[self.probes[i]]  # TODO test
            for j in range(self.num_features):
                feature2 = self.features[j]
                sim = probe_sims[i, j]
                if feature2 in features1:
                    if sim > thr:
                        tp[i] += 1
                    else:
                        fn[i] += 1
                else:
                    if sim > thr:
                        fp[i] += 1
                    else:
                        tn[i] += 1
        return tp, tn, fp, fn

    def score_novice(self, probe_sims):
        # np.random.shuffle(probe_sims)
        # np.random.shuffle(probe_sims.T)

        calc_signals = partial(self.calc_signals, probe_sims)
        sims_mean = np.asscalar(np.mean(probe_sims))
        self.novice_score = calc_balanced_accuracy(calc_signals, sims_mean)
