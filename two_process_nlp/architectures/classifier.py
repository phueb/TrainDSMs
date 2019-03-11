import numpy as np
import os
import tensorflow as tf
from itertools import product
import time

from two_process_nlp import config
from two_process_nlp.job_utils import w2e_to_sims

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Params:
    standardize = [True]
    mb_size = [64]
    beta = [0.0]  # 0.0 is best
    learning_rate = [0.1]
    num_hiddens = [0]  # 0 is best
    neg_pos_ratio = [1.0]


name = 'classifier'


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
    assert len(set(evaluator.col_words)) == len(evaluator.col_words)  # using .index() method below
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
        x2_test += [[evaluator.col_words.index(c) for c in candidates]]
        eval_sims_mat_row_ids_test.append(eval_sims_mat_row_id)
    # train
    num_skipped = 0
    for n, (row_words, candidate_rows, row_word_ids_chunk) in enumerate(zip(
            np.array_split(evaluator.row_words, config.Eval.num_folds),
            np.array_split(evaluator.eval_candidates_mat, config.Eval.num_folds),
            np.array_split(row_word_ids, config.Eval.num_folds))):
        if n != fold_id:
            for probe, candidates, eval_sims_mat_row_id in zip(row_words, candidate_rows, row_word_ids_chunk):
                for p, c in product([probe], candidates):
                    if (p, c) in test_pairs:
                        num_skipped += 1
                        continue
                    if c in evaluator.probe2relata[p] or evaluator.check_negative_example(trial, p, c):
                        x1_train.append(w2e[p])
                        x2_train.append(evaluator.col_words.index(c))
                        y_train.append(1.0 if c in evaluator.probe2relata[p] else 0.0)
    x1_train = np.vstack(x1_train)
    x2_train = np.array(x2_train)
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
    num_outputs = len(evaluator.col_words)

    # smart weight init
    sims_mat = w2e_to_sims(w2e, evaluator.row_words, evaluator.col_words)
    embed_mat = np.vstack([w2e[w] for w in evaluator.row_words])
    wy_init = np.zeros((embed_size, num_outputs))
    for n, sims_mat_col in enumerate(sims_mat.T):
        x, res, rank, s = np.linalg.lstsq(embed_mat, sims_mat_col, rcond=None)
        wy_init[:, n] = x

    class Graph:
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):  # process 1 architectures are typically faster on CPU
                # placeholders
                x1 = tf.placeholder(tf.float32, shape=(None, embed_size), name='x1')
                x2 = tf.placeholder(tf.int32, shape=None, name='output_ids')  # (num outputs to compute)
                y = tf.placeholder(tf.float32, shape=None, name='y')  # (batch size, num outputs to compute)
                # forward
                with tf.name_scope('hidden'):
                    wx = tf.get_variable('wx', shape=[embed_size, trial.params.num_hiddens],
                                         dtype=tf.float32)
                    bx = tf.Variable(tf.zeros([trial.params.num_hiddens]))
                    hidden = tf.nn.tanh(tf.matmul(x1, wx) + bx)
                with tf.name_scope('logits'):
                    if trial.params.num_hiddens > 0:
                        wy = tf.get_variable('wy', shape=(trial.params.num_hiddens, num_outputs), dtype=tf.float32)
                        by = tf.Variable(tf.zeros([num_outputs]))
                        # select cols of wy and elements of by
                        wy_cols = tf.gather(wy, x2, axis=1)
                        by_cols = tf.gather(by, x2, axis=0)
                        logits = tf.reduce_sum(tf.multiply(hidden, tf.transpose(wy_cols)), axis=1)
                    else:
                        init = tf.constant_initializer(wy_init)  # ensures expert performance starts at novice-level
                        wy = tf.get_variable('wy', shape=[embed_size, num_outputs],  dtype=tf.float32, initializer=init)
                        by = tf.Variable(tf.zeros([num_outputs]))
                        # select cols of wy and elements of by
                        wy_cols = tf.gather(wy, x2, axis=1)
                        by_cols = tf.gather(by, x2, axis=0)
                        # pairwise vector-dot-product: calc dot-product for each row in x1 and column in wy_cols
                        logits = tf.reduce_sum(tf.multiply(x1, tf.transpose(wy_cols)), axis=1)
                # loss
                pred_cos = tf.nn.sigmoid(logits)  # gives same ba as tanh
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
                loss_no_reg = tf.reduce_mean(cross_entropy, name='xentropy_mean')
                regularizer = tf.nn.l2_loss(wx) + tf.nn.l2_loss(wy)
                loss = tf.reduce_mean((1 - trial.params.beta) * loss_no_reg +
                                      trial.params.beta * regularizer)
                # optimizer  (adagrad is much better than sgd)
                optimizer = tf.train.AdagradOptimizer(learning_rate=trial.params.learning_rate)
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
            # x1_test is 3d, where each 2d slice is a test-set-size batch of embeddings
            cosines = []
            for x1_mat, x2, eval_sims_mat_row_id in zip(x1_test, x2_test, eval_sims_mat_row_ids_test):
                cos = graph.sess.run(graph.pred_cos, feed_dict={graph.x1: x1_mat,
                                                                graph.x2: x2})
                cosines.append(cos)
                eval_sims_mat_row = cos
                trial.results.eval_sims_mats[eval_id][eval_sims_mat_row_id, :] = eval_sims_mat_row
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
        # train
        graph.sess.run([graph.step], feed_dict={graph.x1: x1_batch, graph.x2: x2_batch, graph.y: y_batch})


def train_expert_on_test_fold(evaluator, trial, graph, data, fold_id):  # TODO leave this for analyses
    raise NotImplementedError
