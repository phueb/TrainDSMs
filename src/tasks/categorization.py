import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization
import pyprind

from src import config
from src.figs import make_categorizer_figs


class Trial(object):
    def __init__(self, name, num_probes, num_cats):
        self.name = name
        # accuracy
        self.train_acc_trajs = np.zeros((config.Categorization.num_evals,
                                         config.Categorization.num_folds))
        self.test_acc_trajs = np.zeros((config.Categorization.num_evals,
                                        config.Categorization.num_folds))
        # softmax
        self.train_softmax_probs = np.zeros((num_probes,
                                             config.Categorization.num_evals,
                                             config.Categorization.num_folds))
        self.test_softmax_probs = np.zeros((num_probes,
                                            config.Categorization.num_evals,
                                            config.Categorization.num_folds))
        self.trained_test_softmax_probs = np.zeros((num_probes,
                                                    config.Categorization.num_evals,
                                                    config.Categorization.num_folds))
        self.x_mat = np.zeros((num_cats,
                               config.Categorization.num_evals,
                               config.Categorization.num_folds))
        self.cms = []  # confusion matrix (1 per fold)


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
        self.trials = None  # each result is a class with many attributes
        self.novice_probe_results = []  # for plotting

    @property
    def row_words(self):  # used to build sims
        return self.probes

    @property
    def col_words(self):
        return self.probes

    def load_data(self):
        p = config.Dirs.tasks / self.name / '{}_{}.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
        both = np.loadtxt(p, dtype='str')
        np.random.shuffle(both)
        probes, cats = both.T
        return probes.tolist(), cats.tolist()

    def make_data(self, w2e, w2freq, fold_id, shuffled):
        # train/test split (separately for each category)
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        train_probes = []
        test_probes = []
        for cat, cat_probes in self.cat2probes.items():
            cat_probes = self.cat2probes[cat].copy()
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
        # repeat train samples proportional to corpus frequency - must happen AFTER split
        if config.Categorization.log_freq:
            probe2logf = {probe: np.log(w2freq[probe]).astype(np.int) for probe in self.probes}
            x_train = np.repeat(x_train, [probe2logf[p] for p in train_probes], axis=0)
            y_train = np.repeat(y_train, [probe2logf[p] for p in train_probes], axis=0)
        # shuffle x-y mapping
        if shuffled:
            print('Shuffling category assignment')
            np.random.shuffle(y_train)
        return x_train, np.squeeze(y_train), x_test, np.squeeze(y_test), train_probes, test_probes

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
                    softmax = tf.nn.softmax(logits)
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
                    loss_no_reg = tf.reduce_mean(cross_entropy, name='xentropy_mean')
                    regularizer = tf.nn.l2_loss(wx) + tf.nn.l2_loss(wy)
                    loss = tf.reduce_mean((1 - config.Categorization.beta) * loss_no_reg +
                                          config.Categorization.beta * regularizer)
                    optimizer = tf.train.AdadeltaOptimizer(learning_rate=config.Categorization.learning_rate)
                    step = optimizer.minimize(loss)
                with tf.device('/cpu:0'):
                    correct = tf.nn.in_top_k(logits, y, 1)
                    num_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
                # session
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
        return Graph()

    @staticmethod
    def generate_random_train_batches(x, y, num_probes, num_steps):
        random_choices = np.random.choice(num_probes, config.Categorization.mb_size * num_steps)
        row_ids_list = np.split(random_choices, num_steps)
        for n, row_ids in enumerate(row_ids_list):
            assert len(row_ids) == config.Categorization.mb_size
            x_batch = x[row_ids]
            y_batch = y[row_ids]
            yield n, x_batch, y_batch

    def train_expert_on_test_fold(self, graph, trial, x_test, y_test, fold_id, test_probes):
        num_test_probes = len(x_test)
        num_train_steps = num_test_probes // config.Categorization.mb_size * config.Categorization.num_epochs
        eval_interval = num_train_steps // config.Categorization.num_evals
        eval_steps = np.arange(0, num_train_steps + eval_interval,
                               eval_interval)[:config.Categorization.num_evals].tolist()  # equal sized intervals
        print('Training on test data to collect number of eval steps to criterion for each probe')
        print('Test data size: {:,}'.format(num_test_probes))
        # training and eval
        for step, x_batch, y_batch in self.generate_random_train_batches(x_test, y_test,
                                                                         num_test_probes, num_train_steps):
            if step in eval_steps:
                eval_id = eval_steps.index(step)
                # test softmax probs
                softmax = graph.sess.run(graph.softmax, feed_dict={graph.x: x_test, graph.y: y_test})
                for p, correct_cat_prob in zip(test_probes, softmax[np.arange(num_test_probes), y_test]):
                    trial.trained_test_softmax_probs[self.probes.index(p), eval_id, fold_id] = correct_cat_prob
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

    def train_expert_on_train_fold(self, graph, trial, data, fold_id):
        x_train, y_train, x_test, y_test, train_probes, test_probes = data
        num_train_probes, num_test_probes = len(x_train), len(x_test)
        num_train_steps = num_train_probes // config.Categorization.mb_size * config.Categorization.num_epochs
        eval_interval = num_train_steps // config.Categorization.num_evals
        eval_steps = np.arange(0, num_train_steps + eval_interval,
                               eval_interval)[:config.Categorization.num_evals].tolist()  # equal sized intervals
        print('Train data size: {:,} | Test data size: {:,}'.format(num_train_probes, num_test_probes))
        # training and eval
        ys = []
        for step, x_batch, y_batch in self.generate_random_train_batches(x_train, y_train,
                                                                         num_train_probes, num_train_steps):
            if step in eval_steps:
                eval_id = eval_steps.index(step)
                # train softmax probs
                softmax = graph.sess.run(graph.softmax, feed_dict={graph.x: x_train, graph.y: y_train})
                for p, correct_cat_prob in zip(train_probes, softmax[np.arange(num_train_probes), y_train]):
                    trial.train_softmax_probs[self.probes.index(p), eval_id, fold_id] = correct_cat_prob
                # test softmax probs
                softmax = graph.sess.run(graph.softmax, feed_dict={graph.x: x_test, graph.y: y_test})
                for p, correct_cat_prob in zip(test_probes, softmax[np.arange(num_test_probes), y_test]):
                    trial.test_softmax_probs[self.probes.index(p), eval_id, fold_id] = correct_cat_prob
                # train accuracy
                num_correct = graph.sess.run(graph.num_correct, feed_dict={graph.x: x_train, graph.y: y_train})
                train_acc = num_correct / float(num_train_probes)
                trial.train_acc_trajs[eval_id, fold_id] = train_acc
                # test accuracy
                num_correct = graph.sess.run(graph.num_correct, feed_dict={graph.x: x_test, graph.y: y_test})
                test_acc = num_correct / float(num_test_probes)
                trial.test_acc_trajs[eval_id, fold_id] = test_acc
                # keep track of number of samples in each category
                trial.x_mat[:, eval_id, fold_id] = [ys.count(cat_id) for cat_id in range(self.num_cats)]
                ys = []
                print('step {:>6,}/{:>6,} |Train Acc={:.2f} |Test Acc={:.2f}'.format(
                    step,
                    num_train_steps - 1,
                    train_acc,
                    test_acc))
            # train
            graph.sess.run([graph.step], feed_dict={graph.x: x_batch, graph.y: y_batch})
            ys += y_batch.tolist()  # collect ys for each eval


            # TODO change to relation network

    def save_figs(self, embedder):
        for trial in self.trials:
            # aggregate over folds
            average_x_mat = np.sum(trial.x_mat, axis=2)
            average_cm = np.sum(trial.cms, axis=0)
            # make average accuracy trajectories - careful not to take mean over arrays with zeros
            train_no_zeros = np.where(trial.train_softmax_probs != 0, trial.train_softmax_probs, np.nan)  # zero to nan
            test_no_zeros = np.where(trial.test_softmax_probs != 0, trial.test_softmax_probs, np.nan)
            trained_test_no_zeros = np.where(trial.trained_test_softmax_probs != 0, trial.trained_test_softmax_probs, np.nan)
            train_softmax_traj = np.nanmean(train_no_zeros, axis=(0, 2))
            test_softmax_traj = np.nanmean(test_no_zeros, axis=(0, 2))
            train_acc_traj = trial.train_acc_trajs.mean(axis=1)
            test_acc_traj = trial.test_acc_trajs.mean(axis=1)
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
                    if softmax_prob > config.Categorization.softmax_criterion:
                        cat2train_evals_to_criterion[cat].append(n)
                        break
                else:
                    cat2train_evals_to_criterion[cat].append(config.Categorization.num_evals)
                # test
                for n, softmax_prob in enumerate(test_row):
                    if softmax_prob > config.Categorization.softmax_criterion:
                        cat2test_evals_to_criterion[cat].append(n)
                        break
                else:
                    cat2test_evals_to_criterion[cat].append(config.Categorization.num_evals)
                # trained test (test probes which have been trained after training on train probes completed)
                for n, softmax_prob in enumerate(trained_test_row):
                    if softmax_prob > config.Categorization.softmax_criterion:
                        cat2trained_test_evals_to_criterion[cat].append(n)
                        break
                else:
                    cat2trained_test_evals_to_criterion[cat].append(config.Categorization.num_evals)
            # novice vs expert by probe
            novice_results_by_probe = self.novice_probe_results
            expert_results_by_probe = trial.expert_probe_results
            # novice vs expert by cat
            cat2novice_result = {cat: [] for cat in self.cats}
            cat2expert_result = {cat: [] for cat in self.cats}
            for cat, nov_acc, exp_acc in zip(self.probe_cats, novice_results_by_probe, expert_results_by_probe):
                cat2novice_result[cat].append(nov_acc)
                cat2expert_result[cat].append(exp_acc)
            novice_results_by_cat = [np.mean(cat2novice_result[cat]) for cat in self.cats]
            expert_results_by_cat = [np.mean(cat2expert_result[cat]) for cat in self.cats]
            # feature diagnosticity about category membership
            probe_embed_mat = np.zeros((self.num_probes, embedder.dim1))
            for n, p in enumerate(self.probes):
                probe_embed_mat[n] = embedder.w2e[p]
            true_col = [True for p in self.probes]
            false_col = [False for p in self.probes]
            feature_diagnosticity_mat = np.zeros((self.num_cats, embedder.dim1))
            pbar = pyprind.ProgBar(embedder.dim1)
            print('Making feature_diagnosticity_mat...')
            for col_id, col in enumerate(probe_embed_mat.T):
                pbar.update()
                for cat_id, cat in enumerate(self.cats):
                    target_col = [True if p in self.cat2probes[cat] else False for p in self.probes]
                    last_f1 = 0.0
                    for thr in np.linspace(np.min(col), np.max(col), num=config.Categorization.num_diagnosticity_steps):
                        thr_col = col > thr
                        tp = np.sum((thr_col == target_col) & (thr_col == true_col))   # tp
                        fp = np.sum((thr_col != target_col) & (thr_col == true_col))   # fp
                        fn = np.sum((thr_col != target_col) & (thr_col == false_col))  # fn
                        f1 = (2 * tp) / (2 * tp + fp + fn)
                        if f1 > last_f1:
                            feature_diagnosticity_mat[cat_id, col_id] = f1
                            last_f1 = f1
            # figs
            for fig, fig_name in make_categorizer_figs(feature_diagnosticity_mat,
                                                       train_acc_traj,
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
                                                       self.cats):
                p = config.Dirs.runs / embedder.time_of_init / self.name / '{}_{}.png'.format(fig_name, trial.name)
                if not p.parent.exists():
                    p.parent.mkdir(parents=True)
                fig.savefig(str(p))
                print('Saved {} to {}'.format(fig_name, p))

    def train_and_score_expert(self, embedder):
        self.trials = []  # need to flush trials (because multiple embedders)
        bools = [False, True] if config.Categorization.run_shuffled else [False]
        for shuffled in bools:
            trial = Trial(name='shuffled' if shuffled else '',
                          num_probes=self.num_probes, num_cats=self.num_cats)
            for fold_id in range(config.Categorization.num_folds):
                # train
                print('Fold {}/{}'.format(fold_id + 1, config.Categorization.num_folds))
                print('Training categorization expert {}...'.format(
                    'with shuffled in-out mapping' if shuffled else ''))
                graph = self.make_classifier_graph(embedder.dim1)
                data = self.make_data(embedder.w2e, embedder.w2freq, fold_id, shuffled)
                x_test = data[2]
                y_test = data[3]
                test_probes = data[5]
                self.train_expert_on_train_fold(graph, trial, data, fold_id)
                # add confusion mat to trial (only includes data for current fold - must be summed over folds)
                logits = graph.sess.run(graph.logits,
                                        feed_dict={graph.x: x_test, graph.y: y_test}).astype(np.int)
                y_pred = np.argmax(logits, axis=1).astype(np.int)
                cm = np.zeros((self.num_cats, self.num_cats))
                for yt, yp in zip(y_test, y_pred):
                    cm[yt, yp] += 1
                trial.cms.append(cm)
                # collect data about steps to criterion
                self.train_expert_on_test_fold(graph, trial, x_test, y_test, fold_id, test_probes)
            trial.expert_probe_results = trial.test_softmax_probs[:, -1, :].mean(axis=1)  # 2d after slicing
            self.trials.append(trial)
        # expert_score
        mean_test_acc_traj = self.trials[0].test_acc_trajs.mean(axis=1)
        best_eval_id = np.argmax(mean_test_acc_traj)
        expert_score = mean_test_acc_traj[best_eval_id]
        print('Expert score={:.2f} (at eval step {})'.format(expert_score, best_eval_id + 1))
        return expert_score

    def score_novice(self, probe_sims, probe_cats=None, metric='ba'):
        if probe_cats is None:
            probe_cats = []
            for p, cat in zip(self.probes, self.probe_cats):
                probe_cats.append(cat)
        assert len(probe_cats) == len(self.probes) == len(probe_sims)

        def calc_p_and_r(thr):
            num_probes = len(probe_sims)
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
                        sim = probe_sims[i, j]
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

        def calc_probes_fs(thr, mean=True):
            precision, recall = calc_p_and_r(thr)
            probe_fs_list = 2 * (precision * recall) / (precision + recall)  # f1-score
            if mean:
                return np.mean(probe_fs_list)
            else:
                return probe_fs_list

        def calc_probes_ba(thr, mean=True):
            precision, recall = calc_p_and_r(thr)
            probe_ba_list = (precision + recall) / 2  # balanced accuracy
            if mean:
                return np.mean(probe_ba_list)
            else:
                return probe_ba_list

        # make thr range
        test_word_sims_mean = np.asscalar(np.mean(probe_sims))
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
        results = fn(best_thr, mean=False)
        self.novice_probe_results = results
        result = np.mean(results)
        return result
