import numpy as np
import tensorflow as tf
from itertools import product
import time

from src import config


class Params:
    shuffled = [False, True]
    margin = [50.0, 100.0]  # must be float, 50 is better than 100 on identification
    mb_size = [64]
    beta = [0.0]  # 0.0 is always better than any beta
    learning_rate = [0.1]
    num_output = [100]  # 100 is better than 30
    # arch-evaluator interaction
    num_epochs = None  # larger for matching vs identification task


name = 'comparator'


def init_results_data(evaluator, eval_data_class):
    """
    add architecture-specific attributes to EvalData class implemented in EvalBase
    """
    assert evaluator is not None
    return eval_data_class


def split_and_vectorize_eval_data(evaluator, trial, w2e, fold_id):
    # split
    x1_train = []
    x2_train = []
    y_train = []
    x1_test = []
    x2_test = []
    eval_sims_mat_row_ids_test = []
    eval_sims_mat_row_ids_train = []
    num_row_words = len(evaluator.row_words)
    row_word_ids = np.arange(num_row_words)  # feed ids explicitly because .index() fails with duplicates
    for n, (row_words, candidate_rows, row_word_ids_chunk) in enumerate(zip(
            np.array_split(evaluator.row_words, config.Eval.num_folds),
            np.array_split(evaluator.eval_candidates_mat, config.Eval.num_folds),
            np.array_split(row_word_ids, config.Eval.num_folds))):
        if n != fold_id:
            for probe, candidates, eval_sims_mat_row_id in zip(row_words, candidate_rows, row_word_ids_chunk):
                for p, c in product([probe], candidates):
                    if config.Eval.only_negative_examples and c in evaluator.probe2relata[p]:  # TODO test
                        continue
                    if c in evaluator.probe2relata[p] or evaluator.check_negative_example(trial, p, c):
                        x1_train.append(w2e[probe])
                        x2_train.append(w2e[c])
                        y_train.append(1 if c in evaluator.probe2relata[p] else 0)
                        eval_sims_mat_row_ids_train.append(eval_sims_mat_row_id)
        else:
            # test data to build chunk of eval_sim_mat
            for probe, candidates, eval_sims_mat_row_id in zip(row_words, candidate_rows, row_word_ids_chunk):
                x1_test += [[w2e[probe]] * len(candidates)]
                x2_test += [[w2e[c] for c in candidates]]
                eval_sims_mat_row_ids_test.append(eval_sims_mat_row_id)
    # check for test to train data leak
    for i in eval_sims_mat_row_ids_train:
        assert i not in eval_sims_mat_row_ids_test
    else:
        print('Number of total  train+test items={}'.format(len(eval_sims_mat_row_ids_train) +
                                                            len(eval_sims_mat_row_ids_test)))
        print('Number of unique train+test items={}'.format(len(np.unique(eval_sims_mat_row_ids_train)) +
                                                            len(np.unique(eval_sims_mat_row_ids_test))))

    x1_train = np.vstack(x1_train)
    x2_train = np.vstack(x2_train)
    y_train = np.array(y_train)
    x1_test = np.array(x1_test)
    x2_test = np.array(x2_test)
    # shuffle x-y mapping
    if trial.params.shuffled:
        print('Shuffling supervisory signal')
        np.random.shuffle(y_train)
    return x1_train, x2_train, y_train, x1_test, x2_test, eval_sims_mat_row_ids_test


def make_graph(evaluator, trial, embed_size):
    assert evaluator is not None   # arbitrary usage of evaluator

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


def train_expert_on_train_fold(evaluator, trial, graph, data, fold_id):
    def gen_batches_in_order(x1, x2, y):
        assert len(x1) == len(x2) == len(y)
        num_rows = len(x1)
        num_adj = num_rows - (num_rows % trial.params.mb_size)
        print('Adjusting for mini-batching: before={} after={} diff={}'.format(num_rows, num_adj, num_rows-num_adj))
        step = 0
        for epoch_id in range(trial.params.num_epochs):
            row_ids = np.random.choice(num_rows, size=num_adj, replace=False)
            # split into batches
            num_splits = num_adj // trial.params.mb_size
            for row_ids in np.split(row_ids, num_splits):
                yield step, x1[row_ids],  x2[row_ids], y[row_ids]
                step += 1

    assert evaluator is not None  # arbitrary usage of evaluator
    assert isinstance(fold_id, int)  # arbitrary usage of fold_id
    x1_train, x2_train, y_train, x1_test, x2_test, eval_sims_mat_row_ids = data
    num_train_probes, num_test_probes = len(x1_train), len(x1_test)
    if num_train_probes < trial.params.mb_size:
        raise RuntimeError('Number of train probes ({}) is less than mb_size={}'.format(
            num_train_probes, trial.params.mb_size))
    num_train_steps = (num_train_probes // trial.params.mb_size) * trial.params.num_epochs
    eval_interval = num_train_steps // config.Eval.num_evals
    eval_steps = np.arange(0, num_train_steps + eval_interval,
                           eval_interval)[:config.Eval.num_evals].tolist()  # equal sized intervals
    print('Train data size: {:,} | Test data size: {:,}'.format(num_train_probes, num_test_probes))
    # training and eval
    start = time.time()
    for step, x1_batch, x2_batch, y_batch in gen_batches_in_order(x1_train, x2_train, y_train):
        if step in eval_steps:
            eval_id = eval_steps.index(step)
            # train loss
            train_loss = graph.sess.run(graph.loss, feed_dict={graph.x1: x1_train,
                                                               graph.x2: x2_train,
                                                               graph.y: y_train})
            # test acc cannot be computed because only partial eval sims mat is available
            for x1_mat, x2_mat, eval_sims_mat_row_id in zip(x1_test, x2_test, eval_sims_mat_row_ids):
                eucd = graph.sess.run(graph.eucd, feed_dict={graph.x1: x1_mat,
                                                             graph.x2: x2_mat})
                eval_sims_mat_row = 1.0 - (eucd / trial.params.margin)
                trial.results.eval_sims_mats[eval_id][eval_sims_mat_row_id, :] = eval_sims_mat_row
            print('step {:>6,}/{:>6,} |Train Loss={:>7.2f} |secs={:.1f} |any nans={}'.format(
                step,
                num_train_steps - 1,
                train_loss,
                time.time() - start,
                np.any(np.isnan(trial.results.eval_sims_mats[eval_id]))))
            start = time.time()
        # train
        graph.sess.run([graph.step], feed_dict={graph.x1: x1_batch, graph.x2: x2_batch, graph.y: y_batch})


def train_expert_on_test_fold(evaluator, trial, graph, data, fold_id):
    raise NotImplementedError

# ////////////////////////////////////////////////////////////////////// figs

def make_trial_figs(self, trial):
    raise NotImplementedError
