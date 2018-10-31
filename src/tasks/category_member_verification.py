import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization

from src import config
from src.figs import make_cat_member_verification_figs
from src.params import make_param2id, ObjectView


class Params:
    shuffled = [False, True]  # TODO save shuffled data separately?
    num_epochs = [500, 1000]
    mb_size = [4, 16]
    num_output = [32, 256]
    margin = [100.0]
    beta = [0.0, 0.2]
    learning_rate = [0.1]
    num_folds = [2]


class Trial(object):
    def __init__(self, params_id, params, num_probes):
        self.params_id = params_id
        self.params = params
        #
        self.train_acc_trajs = np.zeros((config.Task.num_evals, self.params.num_folds))
        self.test_probe_sims = [np.zeros((num_probes, num_probes)) for _ in range(config.Task.num_evals)]


class CatMEmberVer:
    def __init__(self, cat_type):
        self.name = '{}_cat_mem_ver'.format(cat_type)
        self.cat_type = cat_type
        self.probes, self.probe_cats = self.load_data()
        self.cats = sorted(set(self.probe_cats))
        self.cat2probes = {cat: [p for p, c in zip(self.probes, self.probe_cats) if c == cat] for cat in self.cats}
        self.num_probes = len(self.probes)
        self.num_cats = len(self.cats)
        # evaluation
        self.trials = None  # each result is a class with many attributes

    @property
    def row_words(self):  # used to build sims
        return self.probes

    @property
    def col_words(self):
        return self.probes

    def load_data(self):
        p = config.Dirs.tasks / '{}_categories'.format(self.cat_type) / '{}_{}.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
        both = np.loadtxt(p, dtype='str')
        np.random.shuffle(both)
        probes, cats = both.T
        return probes.tolist(), cats.tolist()

    def make_data(self, trial, w2e, fold_id):  # TODO make candidates_mat like in nym_detection
        # train/test split (separately for each category)
        x1_train = []
        x2_train = []
        y_train = []
        x1_test = []
        x2_test = []
        test_probe_ids = []
        probes_copy = self.probes[:]
        for cat, cat_probes in self.cat2probes.items():
            cat_members = np.roll(cat_probes, 1)
            np.random.shuffle(probes_copy)
            distractors = [p for p in probes_copy if p not in cat_probes][:len(cat_probes)]
            for n, (probes_in_fold, members_in_fold, distractors_in_fold) in enumerate(zip(
                    np.array_split(cat_probes, trial.params.num_folds),
                    np.array_split(cat_members, trial.params.num_folds),
                    np.array_split(distractors, trial.params.num_folds))):
                if n != fold_id:
                    x1_train += [w2e[p] for p in probes_in_fold] + [w2e[p] for p in probes_in_fold]
                    x2_train += [w2e[n] for n in members_in_fold] + [w2e[d] for d in distractors_in_fold]
                    y_train += [1] * len(probes_in_fold) + [0] * len(probes_in_fold)
                else:
                    # test data to build chunk of sim matrix as input to balanced accuracy algorithm
                    x1_test += [[w2e[p]] * self.num_probes for p in probes_in_fold]
                    x2_test += [[w2e[p] for p in self.probes] for _ in probes_in_fold]
                    test_probe_ids += [self.probes.index(p) for p in probes_in_fold]
        x1_train = np.vstack(x1_train)
        x2_train = np.vstack(x2_train)
        y_train = np.array(y_train)
        x1_test = np.array(x1_test)
        x2_test = np.array(x2_test)
        assert len(x1_train) == len(x2_train) == len(y_train)
        # shuffle x-y mapping
        if trial.params.shuffled:
            print('Shuffling probe-cat_member mapping')
            np.random.shuffle(y_train)
        return x1_train, x2_train, y_train, x1_test, x2_test, test_probe_ids

    @staticmethod
    def make_graph(trial, embed_size):

        def siamese_leg(x, wy):
            y = tf.matmul(x, wy)
            return y

        class Graph:
            with tf.Graph().as_default():
                with tf.device('/{}:0'.format(config.Task.device)):
                    # placeholders
                    x1 = tf.placeholder(tf.float32, shape=(None, embed_size))
                    x2 = tf.placeholder(tf.float32, shape=(None, embed_size))
                    y = tf.placeholder(tf.float32, [None])
                    # siamese
                    with tf.variable_scope('trial_{}'.format(trial.params_id), reuse=tf.AUTO_REUSE) as scope:
                        wy = tf.get_variable('wy', shape=[embed_size, config.NymMatching.num_output], dtype=tf.float32)
                        o1 = siamese_leg(x1, wy)
                        o2 = siamese_leg(x2, wy)
                    # loss
                    labels_t = y
                    labels_f = tf.subtract(1.0, y)
                    eucd2 = tf.pow(tf.subtract(o1, o2), 2)
                    eucd2 = tf.reduce_sum(eucd2, 1)
                    eucd = tf.sqrt(eucd2 + 1e-6)
                    C = tf.constant(trial.params.margin)
                    # yi*||o1-o2||^2 + (1-yi)*max(0, C-||o1-o2||^2)
                    pos = tf.multiply(labels_t, eucd2)
                    neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2))
                    losses = tf.add(pos, neg)
                    loss_no_reg = tf.reduce_mean(losses)
                    regularizer = tf.nn.l2_loss(wy)
                    loss = tf.reduce_mean((1 - config.Categorization.beta) * loss_no_reg +
                                          trial.params.beta * regularizer)
                    optimizer = tf.train.AdadeltaOptimizer(learning_rate=trial.params.learning_rate)
                    step = optimizer.minimize(loss)
                # session
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
        return Graph()

    def get_best_expert_score(self, trial):
        best_expert_score = 0
        best_eval_id = 0
        for eval_id, sims in enumerate(trial.test_probe_sims):
            expert_score = self.score_novice(sims, verbose=False)
            print('Balanced Accuracy at eval {} is {:.2f}'.format(eval_id + 1, expert_score))
            if expert_score > best_expert_score:
                best_expert_score = expert_score
                best_eval_id = eval_id
        print('Expert score={:.2f} (at eval step {})'.format(best_expert_score, best_eval_id + 1))
        return best_expert_score, best_eval_id

    def train_and_score_expert(self, embedder):
        self.trials = []  # need to flush trials (because multiple embedders reuse task)
        for params_id, (param2id, param2val) in enumerate(make_param2id(Params, stage1=False)):
            trial = Trial(params_id, ObjectView(param2val), self.num_probes)
            print('Training {} expert'.format(self.name))
            for fold_id in range(trial.params.num_folds):
                print('Fold {}/{}'.format(fold_id + 1, trial.params.num_folds))
                graph = self.make_graph(trial, embedder.dim1)
                data = self.make_data(trial, embedder.w2e, fold_id)
                self.train_expert_on_train_fold(trial, graph, data)
            self.get_best_expert_score(trial)
            self.trials.append(trial)
        # expert_score
        # TODO get best across all best scores ?
        raise NotImplementedError

        return best_expert_score

    @staticmethod
    def generate_random_train_batches(x1, x2, y, num_probes, num_steps, mb_size):
        random_choices = np.random.choice(num_probes, mb_size * num_steps)
        row_ids_list = np.split(random_choices, num_steps)
        for n, row_ids in enumerate(row_ids_list):
            assert len(row_ids) == mb_size
            x1_batch = x1[row_ids]
            x2_batch = x2[row_ids]
            y_batch = y[row_ids]
            yield n, x1_batch, x2_batch, y_batch

    def train_expert_on_train_fold(self, trial, graph, data):
        x1_train, x2_train, y_train, x1_test, x2_test, test_probe_ids = data
        num_train_probes, num_test_probes = len(x1_train), len(x1_test)
        num_train_steps = num_train_probes // trial.params.mb_size * trial.params.num_epochs
        eval_interval = num_train_steps // config.Task.num_evals
        eval_steps = np.arange(0, num_train_steps + eval_interval,
                               eval_interval)[:config.Task.num_evals].tolist()  # equal sized intervals
        print('Train data size: {:,} | Test data size: {:,}'.format(num_train_probes, num_test_probes))
        # training and eval
        for step, x1_batch, x2_batch, y_batch in self.generate_random_train_batches(x1_train,
                                                                                    x2_train,
                                                                                    y_train,
                                                                                    num_train_probes,
                                                                                    num_train_steps,
                                                                                    trial.params.mb_size):
            if step in eval_steps:
                eval_id = eval_steps.index(step)
                # train loss - can't evaluate accuracy because training requires 1:1 match vs. non-match
                train_loss = graph.sess.run(graph.loss, feed_dict={graph.x1: x1_train,
                                                                   graph.x2: x2_train,
                                                                   graph.y: y_train})

                # test acc cannot be computed because only partial probe sims mat is available
                for n, (x1_mat, x2_mat, test_probe_id) in enumerate(zip(x1_test, x2_test, test_probe_ids)):
                    eucd = graph.sess.run(graph.eucd, feed_dict={graph.x1: x1_mat,
                                                                 graph.x2: x2_mat})
                    sims_row = 1.0 - (eucd / trial.params.margin)
                    trial.test_probe_sims[eval_id][test_probe_id, :] = sims_row
                test_bal_acc = 'need to complete all folds'
                print('step {:>6,}/{:>6,} |Train Loss={:>7.2f} |Test BalancedAcc={}'.format(
                    step,
                    num_train_steps - 1,
                    train_loss,
                    test_bal_acc))
            # train
            graph.sess.run([graph.step], feed_dict={graph.x1: x1_batch, graph.x2: x2_batch, graph.y: y_batch})

    def save_figs(self, embedder):
        for trial in self.trials:
            # figs
            for fig, fig_name in make_cat_member_verification_figs():
                p = config.Dirs.runs / embedder.time_of_init / self.name / '{}_{}.png'.format(fig_name, trial.params_id)
                if not p.parent.exists():
                    p.parent.mkdir(parents=True)
                fig.savefig(str(p))
                print('Saved {} to {}'.format(fig_name, p))

    def score_novice(self, probe_sims, probe_cats=None, metric='ba', verbose=True):
        if probe_cats is None:
            probe_cats = self.probe_cats
        assert len(probe_cats) == len(probe_sims)

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
        if verbose:
            print('Finding best thresholds between {} and {} using bayesian-optimization...'.format(thr1, thr2))
        gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
        if metric == 'fs':
            fn = calc_probes_fs
        elif metric == 'ba':
            fn = calc_probes_ba
        else:
            raise AttributeError('rnnlab: Invalid arg to "metric".')
        bo = BayesianOptimization(fn, {'thr': (thr1, thr2)}, verbose=verbose)
        bo.explore({'thr': [test_word_sims_mean]})
        bo.maximize(init_points=2, n_iter=config.Task.num_opt_steps,
                    acq="poi", xi=0.001, **gp_params)  # smaller xi: exploitation
        best_thr = bo.res['max']['max_params']['thr']
        # use best_thr
        results = fn(best_thr, mean=False)
        result = np.mean(results)
        return result
