import numpy as np
import tensorflow as tf
from itertools import product
import time

from src import config


class Params:
    shuffled = [False, True]
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
    test_pairs = set()  # prevent train/test leak
    num_row_words = len(evaluator.row_words)
    row_word_ids = np.arange(num_row_words)  # feed ids explicitly because .index() fails with duplicate row_words
    # test - always make test data first to populate test_pairs before making training data
    row_words = np.array_split(evaluator.row_words, config.Eval.num_folds)[fold_id]
    candidate_rows = np.array_split(evaluator.eval_candidates_mat, config.Eval.num_folds)[fold_id]
    row_word_ids_chunk = np.array_split(row_word_ids, config.Eval.num_folds)[fold_id]
    for probe, candidates, eval_sims_mat_row_id in zip(row_words, candidate_rows, row_word_ids_chunk):
        for p, c in product([probe], candidates):
            test_pairs.add((p, c))
            test_pairs.add((c, p))  # crucial to collect both orderings
        #
        x1_test += [[w2e[probe]] * len(candidates)]
        x2_test += [[w2e[c] for c in candidates]]
        eval_sims_mat_row_ids_test.append(eval_sims_mat_row_id)
    # train
    for n, (row_words, candidate_rows, row_word_ids_chunk) in enumerate(zip(
            np.array_split(evaluator.row_words, config.Eval.num_folds),
            np.array_split(evaluator.eval_candidates_mat, config.Eval.num_folds),
            np.array_split(row_word_ids, config.Eval.num_folds))):
        if n != fold_id:
            for probe, candidates, eval_sims_mat_row_id in zip(row_words, candidate_rows, row_word_ids_chunk):
                for p, c in product([probe], candidates):
                    if config.Eval.only_negative_examples:
                        if c in evaluator.probe2relata[p]:  # splitting if statement into two is faster # TODO test
                            continue
                    if (p, c) in test_pairs:
                        continue
                    if c in evaluator.probe2relata[p] or evaluator.check_negative_example(trial, p, c):
                        x1_train.append(w2e[p])
                        x2_train.append(w2e[c])
                        y_train.append(1 if c in evaluator.probe2relata[p] else 0)
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

    def cosine_sim(left, right, eps=1e-12):
        norm_left = tf.sqrt(tf.reduce_sum(tf.square(left), 1) + eps)
        norm_right = tf.sqrt(tf.reduce_sum(tf.square(right), 1) + eps)
        #
        res = tf.reduce_sum(left * right, 1) / (norm_left * norm_right + 1e-10)
        return res

    def siamese_cosine_loss(pred_cos, y, dim0):
        corr_cos = 2 * tf.cast(y, tf.float32) - 1  # converts range [0, 1] to [-1, 1]
        return

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
                corr_cos = 2 * tf.cast(y, tf.float32) - 1  # converts range [0, 1] to [-1, 1]
                pred_cos = cosine_sim(o1, o2)
                mb_size = tf.cast(tf.shape(o1)[0], tf.float32)
                loss_no_reg = tf.nn.l2_loss(corr_cos - pred_cos) / mb_size
                regularizer = tf.nn.l2_loss(wy)
                loss = tf.reduce_mean((1 - trial.params.beta) * loss_no_reg +
                                      trial.params.beta * regularizer)
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=trial.params.learning_rate)
                step = optimizer.minimize(loss)
            # session
            config_proto = tf.ConfigProto()
            config_proto.gpu_options.allow_growth = True
            sess = tf.Session(config=config_proto)
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
            # x1_test and x2_test are 3d, where each 2d slice is a test-set-size batch of embeddings
            cosines = []
            for x1_mat, x2_mat, eval_sims_mat_row_id in zip(x1_test, x2_test, eval_sims_mat_row_ids):
                cos = graph.sess.run(graph.pred_cos, feed_dict={graph.x1: x1_mat,
                                                           graph.x2: x2_mat})
                cosines.append(cos)
                eval_sims_mat_row = cos
                trial.results.eval_sims_mats[eval_id][eval_sims_mat_row_id, :] = eval_sims_mat_row
            print('step {:>9,}/{:>9,} |Train Loss={:>2.2f} |secs={:>2.1f} |any nans={} |mean-cos={:.1f}'.format(
                step,
                num_train_steps - 1,
                train_loss,
                time.time() - start,
                np.any(np.isnan(trial.results.eval_sims_mats[eval_id])),
                np.mean(cosines)))
            start = time.time()
        # train
        graph.sess.run([graph.step], feed_dict={graph.x1: x1_batch, graph.x2: x2_batch, graph.y: y_batch})


def train_expert_on_test_fold(evaluator, trial, graph, data, fold_id):
    raise NotImplementedError

# ////////////////////////////////////////////////////////////////////// figs


def make_trial_figs(self, trial):
    raise NotImplementedError
