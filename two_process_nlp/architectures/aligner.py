import numpy as np
import tensorflow as tf
from itertools import product
import time
import os

from two_process_nlp import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Params:
    standardize = [True]
    mb_size = [64]
    beta = [0.0]  # 0.0 is best
    learning_rate = [0.1]
    num_output = [None]  # 100 is better than 30  but None is best (matches embed_size)
    num_alignment = [30]  # 30 is as good as 100
    neg_pos_ratio = [1.0]  # 1.0 is better than anything higher or lower


name = 'aligner'


def init_results_data(evaluator, eval_data_class):
    """
    add architecture-specific attributes to EvalData class implemented in EvalBase
    """
    assert evaluator is not None
    return eval_data_class


def split_and_vectorize_eval_data(evaluator, trial, w2e, fold_id, shuffled):
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
    test_row_words = np.array_split(evaluator.row_words, config.Eval.num_folds)[fold_id]
    test_candidate_rows = np.array_split(evaluator.eval_candidates_mat, config.Eval.num_folds)[fold_id]
    row_word_ids_chunk = np.array_split(row_word_ids, config.Eval.num_folds)[fold_id]
    for probe, candidates, eval_sims_mat_row_id in zip(test_row_words, test_candidate_rows, row_word_ids_chunk):
        for p, c in product([probe], candidates):
            test_pairs.add((p, c))
            test_pairs.add((c, p))  # crucial to collect both orderings
        #
        x1_test += [[w2e[probe]] * len(candidates)]
        x2_test += [[w2e[c] for c in candidates]]
        eval_sims_mat_row_ids_test.append(eval_sims_mat_row_id)
    # train
    num_skipped = 0
    for n, (train_row_words, train_candidate_rows, row_word_ids_chunk) in enumerate(zip(
            np.array_split(evaluator.row_words, config.Eval.num_folds),
            np.array_split(evaluator.eval_candidates_mat, config.Eval.num_folds),
            np.array_split(row_word_ids, config.Eval.num_folds))):
        if n != fold_id:
            for probe, candidates, eval_sims_mat_row_id in zip(
                    train_row_words, train_candidate_rows, row_word_ids_chunk):
                for p, c in product([probe], candidates):
                    if (p, c) in test_pairs or (c, p) in test_pairs:
                        num_skipped += 1
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
    # console
    print('Num pairs skipped due to occurrence in test={}'.format(num_skipped))
    # shuffle x-y mapping
    if shuffled:
        if config.Eval.verbose:
            print('Shuffling supervisory signal')
        np.random.shuffle(y_train)
    return x1_train, x2_train, y_train, x1_test, x2_test, eval_sims_mat_row_ids_test


def make_graph(evaluator, trial, w2e, embed_size):
    assert evaluator is not None   # arbitrary usage of evaluator
    assert w2e is not None   # arbitrary usage of w2e

    def siamese_leg(x, w):
        y = tf.matmul(x, w)
        return y

    def cosine_sim(left, right, eps=1e-12):
        norm_left = tf.sqrt(tf.reduce_sum(tf.square(left), 1) + eps)
        norm_right = tf.sqrt(tf.reduce_sum(tf.square(right), 1) + eps)
        #
        res = tf.reduce_sum(left * right, 1) / (norm_left * norm_right + 1e-10)
        return res

    class Graph:
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                # placeholders
                x1 = tf.placeholder(tf.float32, shape=(None, embed_size))
                x2 = tf.placeholder(tf.float32, shape=(None, embed_size))
                y = tf.placeholder(tf.float32, [None])
                # forward
                with tf.variable_scope('trial_{}'.format(trial.params_id), reuse=tf.AUTO_REUSE) as scope:
                    if trial.params.num_output is None:
                        if config.Eval.verbose:
                            print('Initializing expert weight matrix with identity matrix.')
                        num_output = embed_size
                        init = tf.constant_initializer(np.eye(num_output))
                    else:
                        if config.Eval.verbose:
                            print('Initializing expert weight matrix with random values.')
                        num_output = trial.params.num_output
                        init = None
                    wy = tf.get_variable('wy', shape=[embed_size, num_output], dtype=tf.float32,
                                         initializer=init)
                    # alignment
                    wa1 = tf.get_variable('wa1', shape=[embed_size, trial.params.num_alignment], dtype=tf.float32)
                    wa2 = tf.get_variable('wa2', shape=[trial.params.num_alignment, embed_size], dtype=tf.float32)
                    combo = tf.add(x1, x2)
                    abstract = tf.matmul(combo, wa1)
                    alignment = tf.matmul(abstract, wa2)
                    #
                    o1 = siamese_leg(x1, wy) + alignment
                    o2 = siamese_leg(x2, wy) + alignment

                # loss
                corr_cos = 2 * tf.cast(y, tf.float32) - 1  # converts range [0, 1] to [-1, 1]
                pred_cos = cosine_sim(o1, o2)
                mb_size = tf.cast(tf.shape(o1)[0], tf.float32)
                loss_no_reg = tf.nn.l2_loss(corr_cos - pred_cos) / mb_size
                regularizer = tf.nn.l2_loss(wy)
                loss = tf.reduce_mean((1 - trial.params.beta) * loss_no_reg +
                                      trial.params.beta * regularizer)
                # optimizer - sgd is not worse or better
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=trial.params.learning_rate)
                step = optimizer.minimize(loss)
            # session
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

    return Graph()


def train_expert_on_train_fold(evaluator, trial, w2e, graph, data, fold_id):
    def gen_batches_in_order(x1, x2, y):
        assert len(x1) == len(x2) == len(y)
        num_rows = len(x1)
        num_adj = num_rows - (num_rows % trial.params.mb_size)
        if config.Eval.verbose:
            print('Adjusting for mini-batching: before={} after={} diff={}'.format(num_rows, num_adj, num_rows-num_adj))
        step = 0
        for epoch_id in range(evaluator.num_epochs):
            row_ids = np.random.choice(num_rows, size=num_adj, replace=False)
            # split into batches
            num_splits = num_adj // trial.params.mb_size
            for row_ids in np.split(row_ids, num_splits):
                yield step, x1[row_ids],  x2[row_ids], y[row_ids]
                step += 1

    assert evaluator is not None  # arbitrary usage of evaluator
    assert isinstance(fold_id, int)  # arbitrary usage of fold_id
    # train size
    x1_train, x2_train, y_train, x1_test, x2_test, eval_sims_mat_row_ids_test = data
    num_train_probes, num_test_probes = len(x1_train), len(x1_test)
    if num_train_probes < trial.params.mb_size:
        raise RuntimeError('Number of train probes ({}) is less than mb_size={}'.format(
            num_train_probes, trial.params.mb_size))
    if config.Eval.verbose:
        print('Train data size: {:,} | Test data size: {:,}'.format(num_train_probes, num_test_probes))
    # eval steps
    num_train_steps = (num_train_probes // trial.params.mb_size) * evaluator.num_epochs
    eval_interval = num_train_steps // config.Eval.num_evals
    eval_steps = np.arange(0, num_train_steps + eval_interval,
                           eval_interval)[:config.Eval.num_evals].tolist()  # equal sized intervals
    # training and eval
    start = time.time()
    for step, x1_batch, x2_batch, y_batch in gen_batches_in_order(x1_train, x2_train, y_train):
        # test
        if step in eval_steps:
            eval_id = eval_steps.index(step)
            # x1_test and x2_test are 3d, where each 2d slice is a test-set-size batch of embeddings
            cosines = []
            for x1_mat, x2_mat, eval_sims_mat_row_id in zip(x1_test, x2_test, eval_sims_mat_row_ids_test):
                cos = graph.sess.run(graph.pred_cos, feed_dict={graph.x1: x1_mat,
                                                                graph.x2: x2_mat})
                cosines.append(cos)
                eval_sims_mat_row = cos
                trial.results.eval_sims_mats[eval_id][eval_sims_mat_row_id, :] = eval_sims_mat_row
            # console
            if config.Eval.verbose:
                train_loss = graph.sess.run(graph.loss, feed_dict={graph.x1: x1_train,
                                                                   graph.x2: x2_train,
                                                                   graph.y: y_train})
                print('step {:>9,}/{:>9,} |Train Loss={:>2.2f} |secs={:>2.1f} |any nans={} |mean-cos={:.1f}'.format(
                    step,
                    num_train_steps - 1,
                    train_loss,
                    time.time() - start,
                    np.any(np.isnan(trial.results.eval_sims_mats[eval_id])),
                    np.mean(cosines)))
            start = time.time()
            # save transformed word embeddings
            if fold_id == 0:
                x1_all = np.vstack((w2e[p] for p in evaluator.row_words))
                x2_all = np.mean(x1_all, axis=0, keepdims=True).repeat(len(x1_all), axis=0)
                # TODO it is incorrect to feed x_all to graph.x2 - feed an average of x_all to graph.x2?
                process2_embeds_mat = graph.sess.run(graph.o1, feed_dict={graph.x1: x1_all, graph.x2: x2_all})
                trial.results.process2_embed_mats[eval_id][:, :] = process2_embeds_mat

        # train
        graph.sess.run([graph.step], feed_dict={graph.x1: x1_batch, graph.x2: x2_batch, graph.y: y_batch})


def train_expert_on_test_fold(evaluator, trial, graph, data, fold_id):  # TODO leave this for analyses
    raise NotImplementedError
