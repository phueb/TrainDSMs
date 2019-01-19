import numpy as np
import os
import tensorflow as tf

from src import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Params:
    beta = [0.0, 0.3]
    num_epochs = [500]
    mb_size = [8]
    learning_rate = [0.1]
    num_hiddens = [32, 256]


name = 'classifier'


def init_results_data(evaluator, eval_data_class):
    """
    add architecture-specific attributes to EvalData class implemented in EvalBase
    """
    num_inputs = evaluator.col_words
    num_outputs = evaluator.row_words
    #
    eval_data_class.train_acc_trajs = np.zeros((config.Eval.num_evals,
                                    config.Eval.num_folds))
    eval_data_class.test_acc_trajs = np.zeros((config.Eval.num_evals,
                                   config.Eval.num_folds))
    eval_data_class.train_softmax_probs = np.zeros((num_inputs,
                                        config.Eval.num_evals,
                                        config.Eval.num_folds))
    eval_data_class.test_softmax_probs = np.zeros((num_inputs,
                                       config.Eval.num_evals,
                                       config.Eval.num_folds))
    eval_data_class.trained_test_softmax_probs = np.zeros((num_inputs,
                                               config.Eval.num_evals,
                                               config.Eval.num_folds))
    eval_data_class.cms = []  # confusion matrix (1 per fold)
    return eval_data_class


def split_and_vectorize_eval_data(evaluator, trial, w2e, fold_id, shuffled):  # don't .index() into row_words  # TODO implement
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
    if shuffled:
        print('Shuffling supervisory signal')
        np.random.shuffle(y_train)
    return x1_train, x2_train, y_train, x1_test, x2_test, eval_sims_mat_row_ids_test


def make_graph(evaluator, trial, embed_size):
    # TODO need a multilabel graph where answers are similarities [-1, 1]
    # TODO this way, the sim matrix can be reconstructed

    num_outputs = len(evaluator.col_words)

    class Graph:
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):  # experts are always faster on CPU
                # placeholders
                x = tf.placeholder(tf.float32, shape=(None, embed_size))
                y = tf.placeholder(tf.int32, shape=None)
                with tf.name_scope('hidden'):
                    wx = tf.get_variable('wx', shape=[embed_size, trial.params.num_hiddens],
                                         dtype=tf.float32)
                    bx = tf.Variable(tf.zeros([trial.params.num_hiddens]))
                    hidden = tf.nn.tanh(tf.matmul(x, wx) + bx)
                with tf.name_scope('logits'):
                    if trial.params.num_hiddens > 0:
                        wy = tf.get_variable('wy', shape=(trial.params.num_hiddens, num_outputs),
                                             dtype=tf.float32)
                        by = tf.Variable(tf.zeros([num_outputs]))
                        logits = tf.matmul(hidden, wy) + by
                    else:
                        wy = tf.get_variable('wy', shape=[embed_size, num_outputs],
                                             dtype=tf.float32)
                        by = tf.Variable(tf.zeros([num_outputs]))
                        logits = tf.matmul(x, wy) + by
                softmax = tf.nn.softmax(logits)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
                loss_no_reg = tf.reduce_mean(cross_entropy, name='xentropy_mean')
                regularizer = tf.nn.l2_loss(wx) + tf.nn.l2_loss(wy)
                loss = tf.reduce_mean((1 - trial.params.beta) * loss_no_reg +
                                      trial.params.beta * regularizer)
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=trial.params.learning_rate)
                step = optimizer.minimize(loss)
            with tf.device('/cpu:0'):
                correct = tf.nn.in_top_k(logits, y, 1)
                num_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
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

    x_train, y_train, x_test, y_test, train_probes, test_probes = data
    num_train_probes, num_test_probes = len(x_train), len(x_test)
    num_train_steps = num_train_probes // trial.params.mb_size * trial.params.num_epochs
    eval_interval = num_train_steps // config.Eval.num_evals
    eval_steps = np.arange(0, num_train_steps + eval_interval,
                           eval_interval)[:config.Eval.num_evals].tolist()  # equal sized intervals
    print('Train data size: {:,} | Test data size: {:,}'.format(num_train_probes, num_test_probes))
    # training and eval
    ys = []
    for step, x_batch, y_batch in generate_random_train_batches(x_train,
                                                                y_train,
                                                                num_train_probes,
                                                                num_train_steps,
                                                                trial.params.mb_size):
        if step in eval_steps:
            eval_id = eval_steps.index(step)
            # train accuracy
            num_correct = graph.sess.run(graph.num_correct, feed_dict={graph.x: x_train, graph.y: y_train})
            train_acc = num_correct / float(num_train_probes)
            trial.results.train_acc_trajs[eval_id, fold_id] = train_acc
            # test accuracy
            num_correct = graph.sess.run(graph.num_correct, feed_dict={graph.x: x_test, graph.y: y_test})
            test_acc = num_correct / float(num_test_probes)
            trial.results.test_acc_trajs[eval_id, fold_id] = test_acc
            print('step {:>6,}/{:>6,} |Train Acc={:.2f} |Test Acc={:.2f}'.format(
                step,
                num_train_steps - 1,
                train_acc,
                test_acc))
        # train
        graph.sess.run([graph.step], feed_dict={graph.x: x_batch, graph.y: y_batch})


def train_expert_on_test_fold(evaluator, trial, graph, data, fold_id):
    raise NotImplementedError
