import numpy as np
import tensorflow as tf


class CatClassifier:  # TODO use different instance for each training?
    def __init__(self, config):
        self.config = config
        self.x, self.y = self.make_input()

    def make_input(self):
        probe_cat_list = [self.hub.probe_store.probe_cat_dict[probe]
                          for probe in self.hub.probe_store.types]
        # make rnn_input
        if self.config.rep_type == 'proto':
            multi_probe_acts_df = self.get_multi_probe_prototype_acts_df()
            x_data = multi_probe_acts_df.values
            y_data = np.zeros(x_data.shape[0])
            for n, cat in enumerate(probe_cat_list):
                y_data[n] = self.hub.probe_store.cats.index(cat)
            probe_freq_list = [np.sum(self.hub.term_part_freq_dict[probe]) for probe in
                               self.hub.probe_store.types]
            probe_freq_list = np.clip(probe_freq_list, 0, num_acts_samples)
            x_data = np.repeat(x_data, probe_freq_list, axis=0)  # repeat according to token frequency
            y_data = np.repeat(y_data, probe_freq_list, axis=0)
        else:
            multi_probe_acts_df = self.get_multi_probe_exemplar_acts_df(num_acts_samples)
            x_data = multi_probe_acts_df.values
            y_data = np.zeros(x_data.shape[0], np.int)
            for n, probe_id in enumerate(multi_probe_acts_df.index.tolist()):
                probe = probe_store.types[probe_id]
                cat = probe_store.probe_cat_dict[probe]
                y_data[n] = probe_store.cats.index(cat)
        if shuffle_cats:
            print('Shuffling category assignment')
            np.random.shuffle(y_data)
        return x_data, y_data

    def run_classifier(self, configs):
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


