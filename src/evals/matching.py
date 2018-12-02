import numpy as np
import tensorflow as tf
from functools import partial
from itertools import product
import time

from src import config
from src.scores import calc_balanced_accuracy
from src.figs import make_matching_figs
from src.evals.base import EvalBase


class Params:
    shuffled = [False, True]
    margin = [50.0, 100.0]
    num_epochs = [100]
    mb_size = [64]
    num_output = [100]
    beta = [0.0]
    learning_rate = [0.1]
    prop_negative = [0.1]  # TODO
    # TODO add option to balance neg:pos

    if any([i > 0.5 for i in prop_negative]):  # TODO test
        raise ValueError('Setting "prop_negative" too high might cause memory error.')


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
        probes, probe_relata = self.load_probes()  # relata can be synonyms, hypernyms, etc.
        relata = sorted(np.unique(np.concatenate(probe_relata)).tolist())
        self.probe2relata = {p: r for p, r in zip(probes, probe_relata)}
        # sims
        self.row_words = probes  # only use these two in methods below (because these are shuffled)
        self.col_words = relata
        # misc
        self.binomial = np.random.binomial

    # ///////////////////////////////////////////// Overwritten Methods START

    def init_results_data(self, trial):
        """
        add task-specific attributes to EvalData class implemented in TaskBase
        """
        res = super().init_results_data(trial)
        res.eval_sims_mats = [np.zeros_like(self.eval_candidates_mat, float)  # same shape as eval_candidates_mat
                               for _ in range(config.Eval.num_evals)]
        return res

    def make_eval_data(self, sims, verbose=False):
        num_row_words = len(self.row_words)
        eval_candidates_mat = np.asarray([self.col_words for _ in range(num_row_words)])
        return eval_candidates_mat

    def check_negative_example(self, trial):
        return bool(self.binomial(n=1, p=trial.params.prop_negative, size=1))

    def split_and_vectorize_eval_data(self, trial, w2e, fold_id):
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
                        if c in self.probe2relata[p] or self.check_negative_example(trial):
                            x1_train.append(w2e[probe])
                            x2_train.append(w2e[c])
                            y_train.append(1 if c in self.probe2relata[p] else 0)
            else:
                # test data to build chunk of eval_sim_mat
                for probe, candidates in zip(eval_probes, candidate_rows):
                    x1_test += [[w2e[probe]] * len(candidates)]
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
                    trial.results.eval_sims_mats[eval_id][row_id, :] = eval_sims_mat_row
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
            gold = self.make_gold()  # make here - because row and col words are shuffled before each rep
            calc_signals = partial(self.calc_signals, eval_sims_mat, gold)
            sims_mean = np.asscalar(np.mean(eval_sims_mat))
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
        probes = []
        probe_relata = []
        with p.open('r') as f:
            for line in f.read().splitlines():
                spl = line.split()
                probes.append(spl[0])
                probe_relata.append(spl[1:])
                if len(probes) > config.Eval.max_num_pairs:
                    print('Excluding all "{}" pairs to protect from memory overloading'.format(spl[0]))
        return probes, probe_relata

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

    @staticmethod
    def calc_signals(probe_sims, gold, thr):  # vectorized algorithm is 20X faster
        predicted = np.zeros_like(probe_sims, int)
        predicted[np.where(probe_sims > thr)] = 1
        tp = float(len(np.where((predicted == gold) & (gold == 1))[0]))
        tn = float(len(np.where((predicted == gold) & (gold == 0))[0]))
        fp = float(len(np.where((predicted != gold) & (gold == 0))[0]))
        fn = float(len(np.where((predicted != gold) & (gold == 1))[0]))
        return tp, tn, fp, fn

    def make_gold(self):  # used for signal detection
        num_rows = len(self.row_words)
        num_cols = len(self.col_words)
        res = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            relata1 = self.probe2relata[self.row_words[i]]
            for j in range(num_cols):
                relatum2 = self.col_words[j]
                if relatum2 in relata1:
                    res[i, j] = 1
        assert int(np.sum(res)) == len(np.concatenate(list(self.probe2relata.values())))
        return res.astype(np.bool)

    def score_novice(self, eval_sims_mat):
        gold = self.make_gold()  # make here - because row and col words are shuffled before each rep
        calc_signals = partial(self.calc_signals, eval_sims_mat, gold)
        sims_mean = np.asscalar(np.mean(eval_sims_mat))
        self.novice_score = calc_balanced_accuracy(calc_signals, sims_mean)