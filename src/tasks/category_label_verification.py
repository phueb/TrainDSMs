import numpy as np
import tensorflow as tf
import pyprind

from src import config
from src.figs import make_categorizer_figs
from src.tasks.base import TaskBase


class Params:
    beta = [0.0]
    num_folds = [2, 6]  # also determines number of examples in test vs. train splits
    num_epochs = [500]  # 500
    mb_size = [8]
    learning_rate = [0.1]
    num_hiddens = [32, 256]


class CatLabelVer(TaskBase):
    def __init__(self, cat_type):
        name = '{}_cat_label_ver'.format(cat_type)
        super().__init__(name, Params)
        #
        self.cat_type = cat_type
        self.probes, self.probe_cats = self.load_training_data()
        self.cats = sorted(set(self.probe_cats))
        self.cat2probes = {cat: [p for p, c in zip(self.probes, self.probe_cats) if c == cat] for cat in self.cats}
        self.num_probes = len(self.probes)
        self.num_cats = len(self.cats)
        # evaluation
        self.novice_probe_results = None  # for plotting TODO test
        # sims
        self.row_words = self.probes
        self.col_words = self.cats  # TODO test

    # ///////////////////////////////////////////// Overwritten Methods START

    def init_eval_data(self, trial):
        """
        add task-specific attributes to EvalData class implemented in TaskBase
        """
        res = super().init_eval_data(trial)
        res.train_acc_trajs = np.zeros((config.Task.num_evals,
                                        trial.params.num_folds))
        res.test_acc_trajs = np.zeros((config.Task.num_evals,
                                       trial.params.num_folds))
        res.train_softmax_probs = np.zeros((self.num_probes,
                                            config.Task.num_evals,
                                            trial.params.num_folds))
        res.test_softmax_probs = np.zeros((self.num_probes,
                                           config.Task.num_evals,
                                           trial.params.num_folds))
        res.trained_test_softmax_probs = np.zeros((self.num_probes,
                                                   config.Task.num_evals,
                                                   trial.params.num_folds))
        res.x_mat = np.zeros((self.num_cats,
                              config.Task.num_evals,
                              trial.params.num_folds))
        res.cms = []  # confusion matrix (1 per fold)
        return res

    def make_data(self, trial, w2e, fold_id):
        # train/test split (separately for each category)
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        train_probes = []
        test_probes = []
        for cat, cat_probes in self.cat2probes.items():
            cat_probes = self.cat2probes[cat].copy()
            for n, probes_in_fold in enumerate(np.array_split(cat_probes, trial.params.num_folds)):
                xs = [w2e[p] for p in probes_in_fold]
                ys = [self.cats.index(cat)] * len(probes_in_fold)
                if n != fold_id:
                    x_train += xs
                    y_train += ys
                    train_probes += probes_in_fold.tolist()
                else:
                    x_test += xs
                    y_test += ys
                    test_probes += probes_in_fold.tolist()
        x_train = np.vstack(x_train)
        y_train = np.vstack(y_train)
        x_test = np.vstack(x_test)
        y_test = np.vstack(y_test)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        assert len(x_train) + len(x_test) == self.num_probes
        assert len(test_probes) == len(x_test)
        assert len(train_probes) == len(x_train)
        # shuffle x-y mapping
        if trial.params.shuffled:
            print('Shuffling category assignment')
            np.random.shuffle(y_train)
        return x_train, np.squeeze(y_train), x_test, np.squeeze(y_test), train_probes, test_probes

    def make_graph(self, trial, embed_size):
        class Graph:
            with tf.Graph().as_default():
                with tf.device('/{}:0'.format(config.Task.device)):
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
                            wy = tf.get_variable('wy', shape=(trial.params.num_hiddens, self.num_cats),
                                                 dtype=tf.float32)
                            by = tf.Variable(tf.zeros([self.num_cats]))
                            logits = tf.matmul(hidden, wy) + by
                        else:
                            wy = tf.get_variable('wy', shape=[embed_size, self.num_cats],
                                                 dtype=tf.float32)
                            by = tf.Variable(tf.zeros([self.num_cats]))
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

    def train_expert_on_train_fold(self, graph, trial, data, fold_id):
        x_train, y_train, x_test, y_test, train_probes, test_probes = data
        num_train_probes, num_test_probes = len(x_train), len(x_test)
        num_train_steps = num_train_probes // trial.params.mb_size * trial.params.num_epochs
        eval_interval = num_train_steps // trial.params.num_evals
        eval_steps = np.arange(0, num_train_steps + eval_interval,
                               eval_interval)[:trial.params.num_evals].tolist()  # equal sized intervals
        print('Train data size: {:,} | Test data size: {:,}'.format(num_train_probes, num_test_probes))
        # training and eval
        ys = []
        for step, x_batch, y_batch in self.generate_random_train_batches(x_train,
                                                                         y_train,
                                                                         num_train_probes,
                                                                         num_train_steps,
                                                                         trial.params.mb_size):
            if step in eval_steps:
                eval_id = eval_steps.index(step)
                # train softmax probs
                softmax = graph.sess.run(graph.softmax, feed_dict={graph.x: x_train, graph.y: y_train})
                for p, correct_cat_prob in zip(train_probes, softmax[np.arange(num_train_probes), y_train]):
                    trial.eval.train_softmax_probs[self.probes.index(p), eval_id, fold_id] = correct_cat_prob
                # test softmax probs
                softmax = graph.sess.run(graph.softmax, feed_dict={graph.x: x_test, graph.y: y_test})
                for p, correct_cat_prob in zip(test_probes, softmax[np.arange(num_test_probes), y_test]):
                    trial.eval.test_softmax_probs[self.probes.index(p), eval_id, fold_id] = correct_cat_prob
                # train accuracy
                num_correct = graph.sess.run(graph.num_correct, feed_dict={graph.x: x_train, graph.y: y_train})
                train_acc = num_correct / float(num_train_probes)
                trial.eval.train_acc_trajs[eval_id, fold_id] = train_acc
                # test accuracy
                num_correct = graph.sess.run(graph.num_correct, feed_dict={graph.x: x_test, graph.y: y_test})
                test_acc = num_correct / float(num_test_probes)
                trial.eval.test_acc_trajs[eval_id, fold_id] = test_acc
                # keep track of number of samples in each category
                trial.eval.x_mat[:, eval_id, fold_id] = [ys.count(cat_id) for cat_id in range(self.num_cats)]
                ys = []
                print('step {:>6,}/{:>6,} |Train Acc={:.2f} |Test Acc={:.2f}'.format(
                    step,
                    num_train_steps - 1,
                    train_acc,
                    test_acc))
            # train
            graph.sess.run([graph.step], feed_dict={graph.x: x_batch, graph.y: y_batch})
            ys += y_batch.tolist()  # collect ys for each eval

    def train_expert_on_test_fold(self, trial, graph, data, fold_id):
        x_train, y_train, x_test, y_test, train_probes, test_probes = data
        num_test_probes = len(x_test)
        num_train_steps = num_test_probes // trial.params.mb_size * trial.params.num_epochs
        eval_interval = num_train_steps // trial.params.num_evals
        eval_steps = np.arange(0, num_train_steps + eval_interval,
                               eval_interval)[:trial.params.num_evals].tolist()  # equal sized intervals
        print('Training on test data to collect number of eval steps to criterion for each probe')
        print('Test data size: {:,}'.format(num_test_probes))
        # training and eval
        for step, x_batch, y_batch in self.generate_random_train_batches(x_test,
                                                                         y_test,
                                                                         num_test_probes,
                                                                         num_train_steps,
                                                                         trial.params.mb_size):
            if step in eval_steps:
                eval_id = eval_steps.index(step)
                # test softmax probs
                softmax = graph.sess.run(graph.softmax, feed_dict={graph.x: x_test, graph.y: y_test})
                for p, correct_cat_prob in zip(test_probes, softmax[np.arange(num_test_probes), y_test]):
                    trial.eval.trained_test_softmax_probs[self.probes.index(p), eval_id, fold_id] = correct_cat_prob
                # accuracy
                num_correct = graph.sess.run(graph.num_correct, feed_dict={graph.x: x_test, graph.y: y_test})
                test_acc = num_correct / float(num_test_probes)
                # keep track of number of samples in each category
                print('step {:>6,}/{:>6,} |Acc={:.2f}'.format(
                    step,
                    num_train_steps - 1,
                    test_acc))
            # train
            graph.sess.run([graph.step], feed_dict={graph.x: x_batch, graph.y: y_batch})

    def get_best_trial_score(self, trial):
        raise NotImplementedError

    def make_trial_figs(self, trial):
        # aggregate over folds
        average_x_mat = np.sum(trial.eval.x_mat, axis=2)
        average_cm = np.sum(trial.eval.cms, axis=0)
        # make average accuracy trajectories - careful not to take mean over arrays with zeros
        train_no_zeros = np.where(trial.eval.train_softmax_probs != 0, trial.eval.train_softmax_probs, np.nan)  # zero to nan
        test_no_zeros = np.where(trial.eval.test_softmax_probs != 0, trial.eval.test_softmax_probs, np.nan)
        trained_test_no_zeros = np.where(trial.eval.trained_test_softmax_probs != 0, trial.eval.trained_test_softmax_probs, np.nan)
        train_softmax_traj = np.nanmean(train_no_zeros, axis=(0, 2))
        test_softmax_traj = np.nanmean(test_no_zeros, axis=(0, 2))
        train_acc_traj = trial.eval.train_acc_trajs.mean(axis=1)
        test_acc_traj = trial.eval.test_acc_trajs.mean(axis=1)
        # make data for criterion fig
        train_tmp = np.nanmean(train_no_zeros, axis=2)  # [num _probes, num_evals]
        test_tmp = np.nanmean(test_no_zeros, axis=2)  # [num _probes, num_evals]
        trained_test_tmp = np.nanmean(trained_test_no_zeros, axis=2)  # [num _probes, num_evals]
        cat2train_evals_to_criterion = {cat: [] for cat in self.cats}
        cat2test_evals_to_criterion = {cat: [] for cat in self.cats}
        cat2trained_test_evals_to_criterion = {cat: [] for cat in self.cats}
        for probe, cat, train_row, test_row, trained_test_row in zip(self.probes, self.probe_cats,
                                                                     train_tmp, test_tmp, trained_test_tmp):
            # train
            for n, softmax_prob in enumerate(train_row):
                if softmax_prob > config.Figs.softmax_criterion:
                    cat2train_evals_to_criterion[cat].append(n)
                    break
            else:
                cat2train_evals_to_criterion[cat].append(config.Task.num_evals)
            # test
            for n, softmax_prob in enumerate(test_row):
                if softmax_prob > config.Figs.softmax_criterion:
                    cat2test_evals_to_criterion[cat].append(n)
                    break
            else:
                cat2test_evals_to_criterion[cat].append(config.Task.num_evals)
            # trained test (test probes which have been trained after training on train probes completed)
            for n, softmax_prob in enumerate(trained_test_row):
                if softmax_prob > config.Figs.softmax_criterion:
                    cat2trained_test_evals_to_criterion[cat].append(n)
                    break
            else:
                cat2trained_test_evals_to_criterion[cat].append(config.Task.num_evals)
        # novice vs expert by probe
        novice_results_by_probe = self.novice_probe_results
        expert_results_by_probe = trial.eval.expert_probe_results
        # novice vs expert by cat
        cat2novice_result = {cat: [] for cat in self.cats}
        cat2expert_result = {cat: [] for cat in self.cats}
        for cat, nov_acc, exp_acc in zip(self.probe_cats, novice_results_by_probe, expert_results_by_probe):
            cat2novice_result[cat].append(nov_acc)
            cat2expert_result[cat].append(exp_acc)
        novice_results_by_cat = [np.mean(cat2novice_result[cat]) for cat in self.cats]
        expert_results_by_cat = [np.mean(cat2expert_result[cat]) for cat in self.cats]

        return make_categorizer_figs(train_acc_traj,
                                     test_acc_traj,
                                     train_softmax_traj,
                                     test_softmax_traj,
                                     average_cm,
                                     np.cumsum(average_x_mat, axis=1),
                                     novice_results_by_cat,
                                     expert_results_by_cat,
                                     novice_results_by_probe,
                                     expert_results_by_probe,
                                     cat2train_evals_to_criterion,
                                     cat2test_evals_to_criterion,
                                     cat2trained_test_evals_to_criterion,
                                     self.cats)

    # ////////////////////////////////////////////// Overwritten Methods END

    def load_training_data(self):
        p = config.Dirs.tasks / '{}_category_labels'.format(self.cat_type) / '{}_{}.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
        both = np.loadtxt(p, dtype='str')
        np.random.shuffle(both)
        probes, probe_cats = both.T
        return probes.tolist(), [c.lower() for c in probe_cats.tolist()]

    @staticmethod
    def generate_random_train_batches(x, y, num_probes, num_steps, mb_size):
        random_choices = np.random.choice(num_probes, mb_size * num_steps)
        row_ids_list = np.split(random_choices, num_steps)
        for n, row_ids in enumerate(row_ids_list):
            assert len(row_ids) == mb_size
            x_batch = x[row_ids]
            y_batch = y[row_ids]
            yield n, x_batch, y_batch

    def score_novice(self, probe_sims, probe_cats=None, metric='ba'):

        raise NotImplementedError  # TODO for each probe, correct categorization is if sim (p, cat label) > all sim (p, non-cat label)
