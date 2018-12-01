import numpy as np
import tensorflow as tf
from functools import partial
import pandas as pd
from itertools import product

from src import config
from src.utils import calc_balanced_accuracy
from src.figs import make_matching_figs
from src.evals.base import EvalBase


class Params:
    shuffled = [False, True]
    margin = [50.0, 100.0]
    num_epochs = [500]
    mb_size = [64]
    num_output = [100]
    beta = [0.0]
    learning_rate = [0.1]
    prop_negative = [0.5]  # proportion of negative pairs to train on  (can dramatically improve performance)


class Matching(EvalBase):
    def __init__(self, data_name1, data_name2=None):
        if data_name2 is not None:
            name = '{}_{}_matching'.format(data_name1, data_name2)  # reversed is okay
        else:
            name = '{}_matching'.format(data_name1)
        super().__init__(name, Params)
        #
        self.data_name1 = data_name1
        self.data_name2 = data_name2
        #
        self.probes, self.probe_bros = self.load_probes()  # "brothers" can be synonyms, hypernyms, etc.
        self.probe2bros = {p: bro for p, bro in zip(self.probes, self.probe_bros)}
        self.bros = np.unique(np.concatenate(self.probe_bros)).tolist()
        self.num_probes = len(self.probes)
        self.num_bros = len(self.bros)
        # sims
        self.row_words = self.probes
        self.col_words = self.bros
        # evaluation
        self.gold = self.make_gold()  # make once to save time - it is a boolean array for signal detection

    # ///////////////////////////////////////////// Overwritten Methods START

    def init_eval_data(self, trial):
        """
        add task-specific attributes to EvalData class implemented in TaskBase
        """
        res = super().init_eval_data(trial)
        res.test_probe_sims = [np.zeros((self.num_probes, self.num_bros))
                               for _ in range(config.Eval.num_evals)]
        return res

    def make_data(self, trial, w2e, fold_id):
        # train/test split
        x1_train = []
        x2_train = []
        y_train = []
        x1_test = []
        x2_test = []
        test_probe_ids = []
        np.random.seed(trial.params_id)  # otherwise randomization is the same in each process
        shuffled_probes = self.probes[:]
        np.random.shuffle(shuffled_probes)  # probes are shuffled when loaded but not between trials
        all_pairs = list(product(shuffled_probes, self.bros))  # produces all probe-brother pairs
        if trial.params.prop_negative > 0.5:
            raise ValueError('Setting "prop_negative" too high might cause memory error.')
        for n, pairs in enumerate(np.array_split(all_pairs, config.Eval.num_folds)):
            if n != fold_id:
                for probe, brother in pairs:
                    is_bro = 1 if brother in self.probe2bros[probe] else 0
                    if is_bro or bool(np.random.binomial(n=1, p=trial.params.prop_negative, size=1)):
                        x1_train.append(w2e[probe])
                        x2_train.append(w2e[brother])
                        y_train.append(is_bro)
            else:
                # test data to build chunk of sim matrix as input to balanced accuracy algorithm
                test_probes = set(list(zip(*pairs))[0])
                for test_probe in test_probes:
                    x1_test += [[w2e[test_probe]] * self.num_bros]
                    x2_test += [[w2e[f] for f in self.bros]]
                    test_probe_ids.append(self.probes.index(test_probe))
        x1_train = np.vstack(x1_train)
        x2_train = np.vstack(x2_train)
        y_train = np.array(y_train)
        x1_test = np.array(x1_test)
        x2_test = np.array(x2_test)
        assert len(x1_train) == len(x2_train) == len(y_train)
        # shuffle x-y mapping
        if trial.params.shuffled:
            print('Shuffling supervisory signal')
            np.random.shuffle(y_train)
        return x1_train, x2_train, y_train, x1_test, x2_test, test_probe_ids

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
        x1_train, x2_train, y_train, x1_test, x2_test, test_probe_ids = data
        num_train_probes, num_test_probes = len(x1_train), len(x1_test)
        num_train_steps = num_train_probes // trial.params.mb_size * trial.params.num_epochs
        eval_interval = num_train_steps // config.Eval.num_evals
        eval_steps = np.arange(0, num_train_steps + eval_interval,
                               eval_interval)[:config.Eval.num_evals].tolist()  # equal sized intervals
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
            print('{} at eval {} is {:.2f}'.format(config.Eval.metric, eval_id + 1, expert_score))
            if expert_score > best_expert_score:
                best_expert_score = expert_score
                best_eval_id = eval_id
        print('Expert score={:.2f} (at eval step {})'.format(best_expert_score, best_eval_id + 1))
        return best_expert_score

    def make_trial_figs(self, trial):
        return make_matching_figs()

    # //////////////////////////////////////////////////// Overwritten Methods END

    def load_probes(self):
        data_dir = '{}/{}'.format(self.data_name1, self.data_name2) if self.data_name2 is not None else self.data_name1
        p = config.Dirs.tasks / data_dir / '{}_{}.txt'.format(
            config.Corpus.name, config.Corpus.num_vocab)
        df = pd.read_csv(p, sep=' ', header=None, error_bad_lines=False)
        if len(df) > config.Eval.max_num_pairs:
            df = df.sample(n=config.Eval.max_num_pairs)
            print('Downsampling number of pairs from {} to {}'.format(len(df), config.Eval.max_num_pairs))
        else:
            df = df.sample(frac=1.0)

        raise NotImplementedError("don't use pd.read_csv - it ignores any column that has index larger than largest col index of first row")

        probes = []
        probe_brothers = []
        for n, row in df.iterrows():
            probes.append(row[0])
            probe_brothers.append([f for f in row[1:] if f is not np.nan])
        return probes, probe_brothers

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

    def calc_signals(self, probe_sims, thr):  # vectorized algorithm is 20X faster
        predicted = np.zeros_like(probe_sims, int)
        predicted[np.where(probe_sims > thr)] = 1
        tp = float(len(np.where((predicted == self.gold) & (self.gold == 1))[0]))
        tn = float(len(np.where((predicted == self.gold) & (self.gold == 0))[0]))
        fp = float(len(np.where((predicted != self.gold) & (self.gold == 0))[0]))
        fn = float(len(np.where((predicted != self.gold) & (self.gold == 1))[0]))
        return tp, tn, fp, fn

    def make_gold(self):  # used for signal detection
        res = np.zeros((self.num_probes, self.num_bros))
        for i in range(self.num_probes):
            bros1 = self.probe2bros[self.probes[i]]
            for j in range(self.num_bros):
                bro2 = self.bros[j]
                if bro2 in bros1:
                    res[i, j] = 1
        assert int(np.sum(res)) == len(np.concatenate(self.probe_bros))
        return res.astype(np.bool)

    def score_novice(self, probe_sims):
        calc_signals = partial(self.calc_signals, probe_sims)
        sims_mean = np.asscalar(np.mean(probe_sims))
        self.novice_score = calc_balanced_accuracy(calc_signals, sims_mean)