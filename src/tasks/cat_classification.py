import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization
from sortedcontainers import SortedDict
from sklearn.model_selection import train_test_split
import pandas as pd

from src import config
from src.figs import make_classifier_figs


class CatClassification:
    def __init__(self, cat_type):
        self.cat_type = cat_type
        self.p2cat = self.make_p2cat()
        self.probes = list(self.p2cat.keys())
        self.cats = list(sorted(self.p2cat.values()))
        assert self.probes == sorted(self.probes)
        assert self.cats == sorted(self.cats)
        self.num_cats = len(self.cats)
        self.num_probes = len(self.probes)
        self.x, self.y, self.logits, self.step, self.correct, self.num_correct, self.sess = self.make_classifier_graph()
        # evaluation
        self.train_acc_traj = []
        self.test_acc_traj = []
        self.ys_per_step = []
        self.train_accs_by_cat = []
        self.test_accs_by_cat = []

    def make_p2cat(self):
        res = SortedDict()
        p = config.Categorization.dir / '{}_categorization.txt'.format(self.cat_type)
        with p.open('r') as f:
            for line in f:
                data = line.strip().strip('\n').split()
                cat = data[0]
                probe = data[1]
                res[probe] = cat
        return res

    def make_classifier_input(self, w2e):
        x = np.array(w2e.values())
        y = np.zeros(x.shape[0], dtype=np.int)
        for n, probe in enumerate(self.probes):
            y[n] = self.p2cat[probe]
        probe_freq_list = [np.sum(self.hub.term_part_freq_dict[probe]) for probe in  # TODO get corpus freq
                           self.probes]
        probe_freq_list = np.clip(probe_freq_list, 0, config.Categorization.num_acts_samples)
        x = np.repeat(x, probe_freq_list, axis=0)  # repeat according to token frequency
        y = np.repeat(y, probe_freq_list, axis=0)
        if config.Categorization.shuffle_cats:
            print('Shuffling category assignment')
            np.random.shuffle(y)
        # split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config.Categorization.test_size)
        # y_test = y_test.astype(np.int)
        # y_train = y_train.astype(np.int)
        return x_train, x_test, y_train, y_test

    def make_classifier_graph(self):
        with tf.Graph().as_default() and tf.device('/gpu:0'):
            # placeholders
            x = tf.placeholder(tf.float32, shape=(None, config.Global.embed_size))
            y = tf.placeholder(tf.int32, shape=(None))
            # hidden ops
            with tf.name_scope('hidden'):
                weights = tf.Variable(tf.truncated_normal([config.Global.embed_size,
                                                           config.Categorization.num_hiddens],
                                                          stddev=1.0 / np.sqrt(config.Global.embed_size)))
                biases = tf.Variable(tf.zeros([config.Categorization.num_hiddens]))
                hidden = tf.nn.tanh(tf.matmul(x, weights) + biases)
            # logits ops
            with tf.name_scope('logits'):
                weights = tf.Variable(tf.truncated_normal([config.Categorization.num_hiddens,
                                                           self.num_cats],
                                                          stddev=1.0 / np.sqrt(config.Categorization.num_hiddens)))
                biases = tf.Variable(tf.zeros([self.num_cats]))
                logits = tf.matmul(hidden, weights) + biases
            # loss ops
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            # training ops
            optimizer = tf.train.AdagradOptimizer(learning_rate=config.Categorization.learning_rate)
            setep = optimizer.minimize(loss)
            # evaluation ops
            correct = tf.nn.in_top_k(logits, y, 1)
            num_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
        # session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        return x, y, logits, setep, correct, num_correct, sess

    def train_expert(self, w2e):
        print('Training categorization expert...')
        # make sure training is done once only
        assert self.train_acc_traj == []
        assert self.test_acc_traj == []
        assert self.ys_per_step == []
        assert self.train_accs_by_cat == []
        assert self.test_accs_by_cat == []
        # data
        x_train, x_test, y_train, y_test = self.make_classifier_input(w2e)
        num_train_examples, num_test_examples = len(x_train), len(x_test)
        num_test_examples = len(x_test)
        max_steps = num_train_examples // config.Categorization.mb_size * config.Categorization.num_epochs
        print('Num train rnn_input: {} | Num test rnn_input: {}'.format(num_train_examples, num_test_examples))

        # batch generator
        def generate_random_train_batches():
            random_choices = np.random.choice(num_train_examples, config.Categorization.mb_size * max_steps)
            row_ids_list = np.split(random_choices, max_steps)
            for step, row_ids in enumerate(row_ids_list):
                assert len(row_ids) == config.Categorization.mb_size
                X_train_batch = x_train[row_ids]
                Y_train_batch = y_train[row_ids]
                yield step, X_train_batch, Y_train_batch

        # function for correct_sum_by_cat
        def make_hca_by_cat(corr, Y):
            hca_by_cat = []
            for cat_id in range(self.num_cats):
                cat_probe_ids = np.where(Y == cat_id)[0]
                num_total_cat_probes = len(cat_probe_ids)
                num_correct_cat_probes = np.sum(corr[cat_probe_ids])
                cat_hca = num_correct_cat_probes / float(num_total_cat_probes) * 100
                hca_by_cat.append(cat_hca)
            return hca_by_cat

        # training and eval
        ys = []
        eval_id = 0
        for step, X_train_batch, Y_train_batch in generate_random_train_batches():
            # train
            self.sess.run([self.step], feed_dict={self.x: X_train_batch, self.y: Y_train_batch})
            if step != 0:
                ys += Y_train_batch.tolist()
                # evaluate
                if step % config.Categorization.num_steps_to_eval == 0:
                    # train accuracy by category
                    correct = self.sess.run(self.correct, feed_dict={self.x: x_train, self.y: y_train})
                    train_acc_by_cat = make_hca_by_cat(correct, y_train)
                    self.train_accs_by_cat.append(train_acc_by_cat)
                    train_acc_unweighted = np.mean(train_acc_by_cat)
                    # test accuracy by category
                    correct = self.sess.run(self.correct, feed_dict={self.x: x_test, self.y: y_test})
                    test_acc_by_cat = make_hca_by_cat(correct, y_test)
                    self.test_accs_by_cat.append(test_acc_by_cat)
                    test_acc_unweighted = np.mean(test_acc_by_cat)
                    # train accuracy
                    num_correct = self.sess.run(self.num_correct, feed_dict={self.x: x_train, self.y: y_train})
                    train_accuracy = num_correct / float(num_train_examples) * 100
                    self.train_acc_traj.append(train_acc_by_cat)
                    # test accuracy
                    num_correct = self.sess.run(self.num_correct, feed_dict={self.x: x_test, self.y: y_test})
                    test_accuracy = num_correct / float(num_test_examples) * 100
                    self.test_acc_traj.append(test_accuracy)
                    # ys_per_step
                    if ys:
                        self.ys_per_step.append(ys)
                    ys = []
                    eval_id += 1
                    print(
                        'step {:,}/{:,} |Train Acc (unw./w.) {:.0f}%/{:.0f}% |Test Acc (unw./w.) {:.0f}%/{:.0f}%'.format(
                            step, max_steps, train_acc_unweighted, train_accuracy, test_acc_unweighted, test_accuracy))

    def save_figs(self, w2e):
        x_train, x_test, y_train, y_test = self.make_classifier_input(w2e)
        dummies = pd.get_dummies(y_test)
        cat_freqs = np.sum(dummies.values, axis=0)
        cat_freqs_mat = np.tile(cat_freqs, (self.num_cats, 1))
        # confusion mat
        logits = self.sess.run(self.logits, feed_dict={self.x: x_test, self.y: y_test}).astype(np.int)
        predicted_y = np.argmax(logits, axis=1).astype(np.int)
        cm = np.zeros((self.num_cats, self.num_cats))
        for y_, py in zip(y_test, predicted_y):
            cm[y_, py] += 1
        cm = np.multiply(cm / cat_freqs_mat, 100).astype(np.int)
        # make x_mat
        ys_per_step_mat = np.asarray(self.ys_per_step).T
        x_mat = np.zeros((self.num_cats, ys_per_step_mat.shape[1]))
        for cat_id in range(self.num_cats):
            cat_id_counts = (ys_per_step_mat == cat_id).sum(axis=0)
            x = np.cumsum(cat_id_counts)
            x_mat[cat_id] = x
        # accuracies by category
        train_acc_trajs = np.array(self.train_accs_by_cat).T  # TODO are categories order alpha or reverse-alpha?
        test_acc_trajs = np.array(self.test_accs_by_cat).T
        # figs
        figs = make_classifier_figs(cm, x_mat, train_acc_trajs, test_acc_trajs)
        # TODO save figs

    def score_expert(self):
        res = self.test_acc_traj[-1]
        return res

    def train_and_score_expert(self, w2e):
        self.train_expert(w2e)
        res = self.score_expert()
        return res

    def score_novice(self, probe_simmat, p2cat=None, metric='ba'):
        if p2cat is None:
            probe_cats = []
            for p, cat in self.p2cat.items():
                probe_cats.append(cat)

        def calc_p_and_r(thr):
            num_probes = len(probe_simmat)
            hits = np.zeros(num_probes, float)
            misses = np.zeros(num_probes, float)
            fas = np.zeros(num_probes, float)
            crs = np.zeros(num_probes, float)
            # calc hits, misses, false alarms, correct rejections
            for i in range(num_probes):
                cat1 = probe_cats[i]
                for j in range(num_probes):
                    if i != j:
                        cat2 = probe_cats[j]
                        sim = probe_simmat[i, j]
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
        probe_simmat_mean = np.asscalar(np.mean(probe_simmat))
        thr1 = max(0.0, round(min(0.9, round(probe_simmat_mean, 2)) - 0.1, 2))  # don't change
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
        bo.explore({'thr': [probe_simmat_mean]})
        bo.maximize(init_points=2, n_iter=config.Categorization.num_opt_steps,
                    acq="poi", xi=0.001, **gp_params)  # smaller xi: exploitation
        best_thr = bo.res['max']['max_params']['thr']
        # use best_thr
        result = fn(best_thr)
        return result
