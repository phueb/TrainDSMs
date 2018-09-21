import numpy as np
import tensorflow as tf

from src import config
from src.figs import make_classifier_figs


class CatClassification:
    def __init__(self, cat_type):
        self.cat_type = cat_type

    def save_figs(self):  # TODO
        figs = make_classifier_figs()

    def train_and_score_expert(self, w2e):  # TODO
        self.train_expert(w2e)
        res = self.score_expert()
        return res

    def make_input(self, x):
        y = np.zeros(x.shape[0])
        for n, probe in enumerate(probes):
            y[n] = p2c[probe]
        probe_freq_list = [np.sum(self.hub.term_part_freq_dict[probe]) for probe in
                           self.hub.probe_store.types]
        probe_freq_list = np.clip(probe_freq_list, 0, config.Categorization.num_acts_samples)
        x = np.repeat(x, probe_freq_list, axis=0)  # repeat according to token frequency
        y = np.repeat(y, probe_freq_list, axis=0)
        if config.Categorization.shuffle_cats:
            print('Shuffling category assignment')
            np.random.shuffle(y)
        return x, y

    def train_expert(self, configs):
        print('Classifying hidden representations...')
        # load_data
        input_dim = x_data.shape[1]
        num_cats = len(list(set(y_data)))
        # train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.3)
        Y_test = Y_test.astype(np.int)
        Y_train = Y_train.astype(np.int)
        num_train_examples, num_test_examples = len(X_train), len(X_test)
        num_test_examples = len(X_test)
        print('Num train rnn_input: {} | Num test rnn_input: {}'.format(num_train_examples, num_test_examples))
        # make eval_step
        max_steps = num_train_examples // mb_size * num_epochs
        if num_steps_to_eval is None:
            num_steps_to_eval = max_steps // num_epochs  # default to evaluating after every epoch
        num_evals = max_steps // configs.num_steps_to_eval
        print('Num eval steps:', num_evals)
        # make cat_freqs for conf_mat
        Y_test_dummies = pd.get_dummies(Y_test)
        cat_freqs = np.sum(Y_test_dummies.values, axis=0)
        cat_freqs_mat = np.tile(cat_freqs, (num_cats, 1))
        with tf.Graph().as_default() and tf.device('/gpu:0'):
            # placeholders
            x = tf.placeholder(tf.float32, shape=(None, input_dim))
            y = tf.placeholder(tf.int32, shape=(None))
            # hidden ops
            with tf.name_scope('hidden'):
                weights = tf.Variable(tf.truncated_normal([input_dim, configs.num_hiddens],
                                                          stddev=1.0 / sqrt(input_dim)), name='weights')
                biases = tf.Variable(tf.zeros([configs.num_hiddens]))
                hidden = tf.nn.tanh(tf.matmul(x, weights) + biases)
            # tf_logits ops
            with tf.name_scope('tf_logits'):
                weights = tf.Variable(tf.truncated_normal([num_hiddens, num_cats],
                                                          stddev=1.0 / sqrt(num_hiddens)), name='weights')
                biases = tf.Variable(tf.zeros([num_cats]))
                tf_logits = tf.matmul(hidden, weights) + biases
            # loss ops
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_logits, labels=y)
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            # training ops
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss)
            # evaluation ops
            tf_correct = tf.nn.in_top_k(tf_logits, y, 1)
            tf_correct_sum = tf.reduce_sum(tf.cast(tf_correct, tf.int32))
        # session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # batch generator
        def generate_random_train_batches():
            random_choices = np.random.choice(num_train_examples, mb_size * max_steps)
            row_ids_list = np.split(random_choices, max_steps)
            for step, row_ids in enumerate(row_ids_list):
                assert len(row_ids) == mb_size
                X_train_batch = X_train[row_ids]
                Y_train_batch = Y_train[row_ids]
                yield step, X_train_batch, Y_train_batch

        # function for correct_sum_by_cat
        def make_hca_by_cat(correct, Y):
            hca_by_cat = []
            for cat_id in range(num_cats):
                cat_probe_ids = np.where(Y == cat_id)[0]
                num_total_cat_probes = len(cat_probe_ids)
                num_correct_cat_probes = np.sum(correct[cat_probe_ids])
                cat_hca = num_correct_cat_probes / float(num_total_cat_probes) * 100
                hca_by_cat.append(cat_hca)
            return hca_by_cat

        # training and eval
        train_hca_traj = []
        test_hca_traj = []
        ys_per_step = []
        ys = []
        eval_id = 0
        train_hca_cat_traj_mat = np.zeros((num_cats, num_evals))
        test_hca_cat_traj_mat = np.zeros((num_cats, num_evals))
        for step, X_train_batch, Y_train_batch in generate_random_train_batches():
            # train
            _, train_loss = sess.run([train_op, loss], feed_dict={x: X_train_batch, y: Y_train_batch})
            if step != 0:
                ys += Y_train_batch.tolist()
                # evaluate
                if step % num_steps_to_eval == 0:
                    # train accuracy by category
                    correct = sess.run(tf_correct, feed_dict={x: X_train, y: Y_train})
                    train_hca_by_cat = make_hca_by_cat(correct, Y_train)
                    train_hca_cat_traj_mat[:, eval_id] = train_hca_by_cat
                    train_hca_unweighted = np.mean(train_hca_by_cat)
                    # test accuracy by category
                    correct = sess.run(tf_correct, feed_dict={x: X_test, y: Y_test})
                    test_hca_by_cat = make_hca_by_cat(correct, Y_test)
                    test_hca_cat_traj_mat[:, eval_id] = test_hca_by_cat
                    test_hca_unweighted = np.mean(test_hca_by_cat)
                    # train accuracy
                    correct_sum = sess.run(tf_correct_sum, feed_dict={x: X_train, y: Y_train})
                    train_hca = correct_sum / float(num_train_examples) * 100
                    train_hca_traj.append(train_hca_by_cat)
                    # test accuracy
                    correct_sum = sess.run(tf_correct_sum, feed_dict={x: X_test, y: Y_test})
                    test_hca = correct_sum / float(num_test_examples) * 100
                    test_hca_traj.append(test_hca)
                    # ys_per_step
                    if ys:
                        ys_per_step.append(ys)
                    ys = []
                    eval_id += 1
                    print(
                        'step {:,}/{:,} |Train Acc (unw./w.) {:.0f}%/{:.0f}% |Test Acc (unw./w.) {:.0f}%/{:.0f}%'.format(
                            step, max_steps, train_hca_unweighted, train_hca, test_hca_unweighted, test_hca))
        # confusion mat
        logits = sess.run(tf_logits, feed_dict={x: X_test, y: Y_test}).astype(np.int)
        predicted_y = np.argmax(logits, axis=1).astype(np.int)
        cm = np.zeros((num_cats, num_cats))
        for y_, py in zip(Y_test.astype(np.int), predicted_y):
            cm[y_, py] += 1
        cm = np.multiply(cm / cat_freqs_mat, 100).astype(np.int)
        # make x_mat
        ys_per_step_mat = np.asarray(ys_per_step).T
        x_mat = np.zeros((num_cats, ys_per_step_mat.shape[1]))
        for cat_id in range(num_cats):
            cat_id_counts = (ys_per_step_mat == cat_id).sum(axis=0)
            x = np.cumsum(cat_id_counts)
            x_mat[cat_id] = x
        # save results
        np.savez(self.hc_data_path,
                 train_hca_cat_traj_mat=train_hca_cat_traj_mat,
                 test_hca_cat_traj_mat=test_hca_cat_traj_mat,
                 cm=cm,
                 x_mat=x_mat)

    @staticmethod
    def score_novice(w2e, metric='ba'):
        def calc_p_and_r(thr):
            hits = np.zeros(hub.probe_store.num_probes, float)
            misses = np.zeros(hub.probe_store.num_probes, float)
            fas = np.zeros(hub.probe_store.num_probes, float)
            crs = np.zeros(hub.probe_store.num_probes, float)
            # calc hits, misses, false alarms, correct rejections
            for i in range(hub.probe_store.num_probes):
                probe1 = hub.probe_store.types[i]
                cat1 = hub.probe_store.probe_cat_dict[probe1]
                for j in range(hub.probe_store.num_probes):
                    if i != j:
                        probe2 = hub.probe_store.types[j]
                        cat2 = hub.probe_store.probe_cat_dict[probe2]
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
        bo.maximize(init_points=2, n_iter=config.Categorization.NUM_BAYES_STEPS,
                    acq="poi", xi=0.001, **gp_params)  # smaller xi: exploitation
        best_thr = bo.res['max']['max_params']['thr']
        # use best_thr
        result = fn(best_thr)
        return result