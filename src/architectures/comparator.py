import numpy as np
import tensorflow as tf
from itertools import product
import time

from src import config


class ComparatorParams:
    shuffled = [False, True]
    margin = [50.0, 100.0]  # must be float and MUST be at least 40 or so
    num_epochs = [100]
    beta = [0.0]
    learning_rate = [0.1]
    mb_size = [4]
    num_output = [100]


class Comparator:  # TODO is this okay to be un-initialized (e.g. tensorflow graph is not duplicated?)
    name = 'comparator'
    params = ComparatorParams

    @staticmethod
    def split_and_vectorize_eval_data(evaluator, trial, w2e, fold_id):
        # split
        x1_train = []
        x2_train = []
        y_train = []
        x1_test = []
        x2_test = []
        eval_sims_mat_row_ids = []
        for n, (eval_probes, candidate_rows) in enumerate(zip(
                np.array_split(evaluator.row_words, config.Eval.num_folds),
                np.array_split(evaluator.eval_candidates_mat, config.Eval.num_folds))):
            if n != fold_id:
                for probe, candidates in zip(eval_probes, candidate_rows):
                    for p, c in product([probe], candidates):
                        if c in evaluator.probe2relata[p] or evaluator.check_negative_example(trial, p, c):
                            x1_train.append(w2e[probe])
                            x2_train.append(w2e[c])
                            y_train.append(1 if c in evaluator.probe2relata[p] else 0)
            else:
                # test data to build chunk of eval_sim_mat
                for probe, candidates in zip(eval_probes, candidate_rows):
                    x1_test += [[w2e[probe]] * len(candidates)]
                    x2_test += [[w2e[c] for c in candidates]]
                    eval_sims_mat_row_ids.append(evaluator.row_words.index(probe))
        x1_train = np.vstack(x1_train)
        x2_train = np.vstack(x2_train)
        y_train = np.array(y_train)
        x1_test = np.array(x1_test)
        x2_test = np.array(x2_test)
        # shuffle x-y mapping
        if trial.params.shuffled:
            print('Shuffling supervisory signal')
            np.random.shuffle(y_train)
        return x1_train, x2_train, y_train, x1_test, x2_test, eval_sims_mat_row_ids

    @staticmethod
    def make_graph(trial, embed_size):

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

    @staticmethod
    def train_expert_on_train_fold(trial, graph, data, fold_id):
        def generate_random_train_batches(x1, x2, y, num_probes, num_steps, mb_size):
            random_choices = np.random.choice(num_probes, mb_size * num_steps)
            row_ids_list = np.split(random_choices, num_steps)
            for n, row_ids in enumerate(row_ids_list):
                assert len(row_ids) == mb_size
                x1_batch = x1[row_ids]
                x2_batch = x2[row_ids]
                y_batch = y[row_ids]
                yield n, x1_batch, x2_batch, y_batch

        assert isinstance(fold_id, int)
        x1_train, x2_train, y_train, x1_test, x2_test, eval_sims_mat_row_ids = data
        num_train_probes, num_test_probes = len(x1_train), len(x1_test)
        num_train_steps = num_train_probes // trial.params.mb_size * trial.params.num_epochs
        eval_interval = num_train_steps // config.Eval.num_evals
        eval_steps = np.arange(0, num_train_steps + eval_interval,
                               eval_interval)[:config.Eval.num_evals].tolist()  # equal sized intervals
        print('Train data size: {:,} | Test data size: {:,}'.format(num_train_probes, num_test_probes))
        # training and eval
        start = time.time()
        for step, x1_batch, x2_batch, y_batch in generate_random_train_batches(x1_train,
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
                for x1_mat, x2_mat, eval_sims_mat_row_id in zip(x1_test, x2_test, eval_sims_mat_row_ids):
                    eucd = graph.sess.run(graph.eucd, feed_dict={graph.x1: x1_mat,
                                                                 graph.x2: x2_mat})
                    eval_sims_mat_row = 1.0 - (eucd / trial.params.margin)
                    trial.results.eval_sims_mats[eval_id][eval_sims_mat_row_id, :] = eval_sims_mat_row
                print('step {:>6,}/{:>6,} |Train Loss={:>7.2f} |secs={:.1f}'.format(
                    step,
                    num_train_steps - 1,
                    train_loss,
                    time.time() - start))
                start = time.time()
            # train
            graph.sess.run([graph.step], feed_dict={graph.x1: x1_batch, graph.x2: x2_batch, graph.y: y_batch})

    @staticmethod
    def train_expert_on_test_fold(trial, graph, data, fold_id):
        raise NotImplementedError

    # ////////////////////////////////////////////////////////////////////// figs

    def make_trial_figs(self, trial):
        raise NotImplementedError
