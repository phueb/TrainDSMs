import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization
import pandas as pd
from cytoolz import itertoolz

from src import config
from src.embedders.base import EmbedderBase
from src.figs import make_categorizer_figs


class Trial(object):  # TODO make this available to all experts?
    def __init__(self, name, num_cats, g, data):
        self.name = name
        self.train_acc_traj = []
        self.test_acc_traj = []
        self.train_accs_by_cat = []
        self.test_accs_by_cat = []
        self.x_mat = np.zeros((num_cats, config.Categorization.num_evals))
        self.g = g
        self.data = data


class Categorization:
    def __init__(self, cat_type):
        self.name = '{}_categorization'.format(cat_type)
        self.cat_type = cat_type
        self.probes, self.probe_cats = self.load_data()
        self.cats = sorted(set(self.probe_cats))
        self.cat2probes = {cat: [p for p, c in zip(self.probes, self.probe_cats) if c == cat] for cat in self.cats}
        self.num_probes = len(self.probes)
        self.num_cats = len(self.cats)
        # evaluation
        self.trials = []  # each result is a class with many attributes

    @property
    def row_words(self):  # used to build sims
        return self.probes

    @property
    def col_words(self):
        return self.probes

    def load_data(self):
        p = config.Dirs.tasks / self.name / '{}_{}.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
        probes, cats = np.loadtxt(p, dtype='str').T
        return probes.tolist(), cats.tolist()

    def make_data(self, w2e, w2freq, fold_id, shuffled):
        # train/test split (separately for each category)
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        train_probes = []
        for cat, cat_probes in self.cat2probes.items():
            cat_probes = self.cat2probes[cat]
            for n, probes_in_fold in enumerate(np.array_split(cat_probes, config.Categorization.num_folds)):
                xs = [w2e[p] for p in probes_in_fold]
                ys = [self.cats.index(cat)] * len(probes_in_fold)
                if n != fold_id:
                    x_train += xs
                    y_train += ys
                    train_probes += probes_in_fold.tolist()
                else:
                    x_test += xs
                    y_test += ys
        x_train = np.vstack(x_train)
        y_train = np.vstack(y_train)
        x_test = np.vstack(x_test)
        y_test = np.vstack(y_test)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        assert len(x_train) + len(x_test) == self.num_probes
        # repeat train samples proportional to corpus frequency - must happen AFTER split
        if config.Categorization.log_freq:
            probe2logf = {probe: np.log(w2freq[probe]).astype(np.int) for probe in self.probes}
            x_train = np.repeat(x_train, [probe2logf[p] for p in train_probes], axis=0)
            y_train = np.repeat(y_train, [probe2logf[p] for p in train_probes], axis=0)
        # shuffle x-y mapping
        if shuffled:
            print('Shuffling category assignment')
            np.random.shuffle(y_train)
        return x_train, np.squeeze(y_train), x_test, np.squeeze(y_test)

    def make_classifier_graph(self, embed_size):
        class Graph:
            with tf.Graph().as_default():
                with tf.device('/{}:0'.format(config.Categorization.device)):
                    # placeholders
                    x = tf.placeholder(tf.float32, shape=(None, embed_size))
                    y = tf.placeholder(tf.int32, shape=None)
                    with tf.name_scope('hidden'):
                        wx = tf.get_variable('wx', shape=[embed_size, config.Categorization.num_hiddens],
                                             dtype=tf.float32)
                        bx = tf.Variable(tf.zeros([config.Categorization.num_hiddens]))
                        hidden = tf.nn.tanh(tf.matmul(x, wx) + bx)
                    with tf.name_scope('logits'):
                        if config.Categorization.num_hiddens > 0:
                            wy = tf.get_variable('wy', shape=(config.Categorization.num_hiddens, self.num_cats),
                                                 dtype=tf.float32)
                            by = tf.Variable(tf.zeros([self.num_cats]))
                            logits = tf.matmul(hidden, wy) + by
                        else:
                            wy = tf.get_variable('wy', shape=[embed_size, self.num_cats],
                                                 dtype=tf.float32)
                            by = tf.Variable(tf.zeros([self.num_cats]))
                            logits = tf.matmul(x, wy) + by
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
                    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.Categorization.learning_rate)  # TODO regularization
                    step = optimizer.minimize(loss)
                with tf.device('/cpu:0'):
                    correct = tf.nn.in_top_k(logits, y, 1)
                    num_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
                # session
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
        return Graph()

    def train_expert(self, trial):
        x_train, y_train, x_test, y_test = trial.data
        num_train_examples, num_test_examples = len(x_train), len(x_test)
        num_test_examples = len(x_test)
        num_train_steps = num_train_examples // config.Categorization.mb_size * config.Categorization.num_epochs
        eval_interval = num_train_steps // config.Categorization.num_evals
        eval_steps = np.arange(0, num_train_steps + eval_interval,
                               eval_interval)[:config.Categorization.num_evals].tolist()  # equal sized intervals
        print('Train data size: {:,} | Test data size: {:,}'.format(num_train_examples, num_test_examples))

        # batch generator
        def generate_random_train_batches():
            random_choices = np.random.choice(num_train_examples, config.Categorization.mb_size * num_train_steps)
            row_ids_list = np.split(random_choices, num_train_steps)
            for step, row_ids in enumerate(row_ids_list):
                assert len(row_ids) == config.Categorization.mb_size
                x_batch = x_train[row_ids]
                y_batch = y_train[row_ids]
                yield step, x_batch, y_batch

        # function for correct_sum_by_cat
        def make_acc_by_cat(corr, y):
            res = []
            for cat_id in range(self.num_cats):
                cat_probe_ids = np.where(y == cat_id)[0]
                num_total_cat_probes = len(cat_probe_ids)
                num_correct_cat_probes = np.sum(corr[cat_probe_ids])
                cat_acc = (num_correct_cat_probes + 1) / (num_total_cat_probes + 1) * 100
                res.append(cat_acc)
            return res

        # training and eval
        ys = []
        for step, x_batch, y_batch in generate_random_train_batches():
            if step in eval_steps:
                # train accuracy by category
                correct = trial.g.sess.run(trial.g.correct, feed_dict={trial.g.x: x_train, trial.g.y: y_train})
                train_acc_by_cat = make_acc_by_cat(correct, y_train)
                trial.train_accs_by_cat.append(train_acc_by_cat)
                train_acc_cat = np.mean(train_acc_by_cat)
                # test accuracy by category
                correct = trial.g.sess.run(trial.g.correct, feed_dict={trial.g.x: x_test, trial.g.y: y_test})
                test_acc_by_cat = make_acc_by_cat(correct, y_test)
                trial.test_accs_by_cat.append(test_acc_by_cat)
                test_acc_cat = np.mean(test_acc_by_cat)
                # train accuracy
                num_correct = trial.g.sess.run(trial.g.num_correct, feed_dict={trial.g.x: x_train, trial.g.y: y_train})
                train_acc_probe = num_correct / float(num_train_examples) * 100
                trial.train_acc_traj.append(train_acc_probe)
                # test accuracy
                num_correct = trial.g.sess.run(trial.g.num_correct, feed_dict={trial.g.x: x_test, trial.g.y: y_test})
                test_acc_probe = num_correct / float(num_test_examples) * 100
                trial.test_acc_traj.append(test_acc_probe)
                # keep track of number of samples in each category
                for cat_id in range(self.num_cats):
                    trial.x_mat[cat_id, eval_steps.index(step)] = ys.count(cat_id)
                ys = []
                print('step {:>6,}/{:>6,} |Train Acc (c/p) {:2.0f}%/{:2.0f}% |Test Acc (c/p) {:2.0f}%/{:2.0f}%'.format(
                    step,
                    num_train_steps - 1,
                    train_acc_cat,
                    train_acc_probe,
                    test_acc_cat,
                    test_acc_probe))
            # train
            trial.g.sess.run([trial.g.step], feed_dict={trial.g.x: x_batch, trial.g.y: y_batch})
            ys += y_batch.tolist()  # collect ys for each eval


            # TODO change to relation network


        return trial

    def save_figs(self, embedder_name):
        for trial in self.trials:
            x_train, y_train, x_test, y_test = trial.data
            dummies = pd.get_dummies(y_test)
            cat_freqs = np.sum(dummies.values, axis=0)
            cat_freqs_mat = np.tile(cat_freqs, (self.num_cats, 1))  # all categories should be in test data
            # confusion mat
            logits = trial.g.sess.run(trial.g.logits, feed_dict={trial.g.x: x_test, trial.g.y: y_test}).astype(np.int)
            predicted_y = np.argmax(logits, axis=1).astype(np.int)
            cm = np.zeros((self.num_cats, self.num_cats))
            for ay, py in zip(y_test, predicted_y):
                cm[ay, py] += 1
            cm = np.multiply(cm / cat_freqs_mat, 100).astype(np.int)
            # accuracies by category
            train_acc_trajs = np.array(trial.train_accs_by_cat).T
            test_acc_trajs = np.array(trial.test_accs_by_cat).T
            # figs
            for fig, fig_name in make_categorizer_figs(cm,
                                                       np.cumsum(trial.x_mat, axis=1),
                                                       train_acc_trajs,
                                                       test_acc_trajs,
                                                       self.cats):
                p = config.Dirs.figs / self.name / '{}_{}_{}.png'.format(fig_name, trial.name, embedder_name)
                if not p.parent.exists():
                    p.parent.mkdir(parents=True)
                fig.savefig(str(p))
                print('Saved "{}" figure to {}'.format(fig_name, config.Dirs.figs / self.name))

    def train_and_score_expert(self, embedder):
        for shuffled in [False, True]:
            name = 'shuffled' if shuffled else ''
            for fold_id in range(config.Categorization.num_folds):

                # TODO a trial should not average over folds (it should not be a single fold)

                trial = Trial(name=name,
                              num_cats=self.num_cats,
                              g=self.make_classifier_graph(embedder.dim1),
                              data=self.make_data(embedder.w2e, embedder.w2freq, fold_id, shuffled))
                print('Fold {}/{}'.format(fold_id + 1, config.Categorization.num_folds))
                print('Training categorization expert {}...'.format(
                    'with shuffled in-out mapping' if shuffled else ''))
                self.train_expert(trial)
                self.trials.append(trial)

        # expert_score
        expert_score = self.trials[0].test_acc_traj[-1]
        return expert_score

    def score_novice(self, test_word_sims, probe_cats=None, metric='ba'):
        if probe_cats is None:
            probe_cats = []
            for p, cat in zip(self.probes, self.probe_cats):
                probe_cats.append(cat)
        assert len(probe_cats) == len(self.probes) == len(test_word_sims)

        def calc_p_and_r(thr):
            num_test_words = len(test_word_sims)
            hits = np.zeros(num_test_words, float)
            misses = np.zeros(num_test_words, float)
            fas = np.zeros(num_test_words, float)
            crs = np.zeros(num_test_words, float)
            # calc hits, misses, false alarms, correct rejections
            for i in range(num_test_words):
                cat1 = probe_cats[i]
                for j in range(num_test_words):
                    if i != j:
                        cat2 = probe_cats[j]
                        sim = test_word_sims[i, j]
                        if cat1 == cat2:
                            if sim > thr:
                                hits[i] += 1
                            else:
                                misses[i] += 1
                        else:
                            if sim > thr:
                                fas[i] += 1
                            else:
                                crs[i] += 1
            avg_probe_recall_list = np.divide(hits + 1, (hits + misses + 1))  # + 1 prevents inf and nan
            avg_probe_precision_list = np.divide(crs + 1, (crs + fas + 1))
            return avg_probe_precision_list, avg_probe_recall_list

        def calc_probes_fs(thr):
            precision, recall = calc_p_and_r(thr)
            probe_fs_list = 2 * (precision * recall) / (precision + recall)  # f1-score
            res = np.mean(probe_fs_list)
            return res

        def calc_probes_ba(thr):
            precision, recall = calc_p_and_r(thr)
            probe_ba_list = (precision + recall) / 2  # balanced accuracy
            res = np.mean(probe_ba_list)
            return res

        # make thr range
        test_word_sims_mean = np.asscalar(np.mean(test_word_sims))
        thr1 = max(0.0, round(min(0.9, round(test_word_sims_mean, 2)) - 0.1, 2))  # don't change
        thr2 = round(thr1 + 0.2, 2)
        # use bayes optimization to find best_thr
        print('Finding best thresholds between {} and {} using bayesian-optimization...'.format(thr1, thr2))
        gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
        if metric == 'fs':
            fn = calc_probes_fs
        elif metric == 'ba':
            fn = calc_probes_ba
        else:
            raise AttributeError('rnnlab: Invalid arg to "metric".')
        bo = BayesianOptimization(fn, {'thr': (thr1, thr2)}, verbose=True)
        bo.explore({'thr': [test_word_sims_mean]})
        bo.maximize(init_points=2, n_iter=config.Categorization.num_opt_steps,
                    acq="poi", xi=0.001, **gp_params)  # smaller xi: exploitation
        best_thr = bo.res['max']['max_params']['thr']
        # use best_thr
        result = fn(best_thr)
        return result
