import numpy as np
import tensorflow as tf
from scipy.stats import binom

from src import config
from src.figs import make_nym_figs


class Trial(object):
    def __init__(self, name):
        self.name = name
        self.train_acc_trajs = np.zeros((config.NymMatching.num_evals, config.NymMatching.num_folds))
        self.test_acc_trajs = np.zeros((config.NymMatching.num_evals, config.NymMatching.num_folds))


class NymDetection:
    def __init__(self, nym_type):
        self.name = '{}_detection'.format(nym_type)
        self.nym_type = nym_type
        self.probes, self.syns, self.ants = self.load_data()
        self.num_probes = len(self.probes)
        self.test_candidates_mat = None  # for evaluation
        # evaluation
        self.trials = None
        # sims
        self.row_words = self.probes
        self.col_words = sorted(set(self.syns + self.ants + self.probes))

    def load_data(self):
        p = config.Dirs.tasks / 'nyms' / '{}_{}.txt'.format(
            config.Corpus.name, config.Corpus.num_vocab)
        print('Loading {}'.format(p))
        loaded = np.loadtxt(p, dtype='str')
        np.random.shuffle(loaded)
        probes, syns, ants = loaded.T
        # remove duplicate nyms
        if config.NymMatching.remove_duplicate_nyms:
            print('Removing dulicate nyms')
            keep_ids = []
            syn_set = set()
            ant_set = set()
            num_triplets = len(loaded)
            for i in range(num_triplets):
                if syns[i] not in syn_set and ants[i] not in ant_set:
                    keep_ids.append(i)
                syn_set.add(syns[i])
                ant_set.add(ants[i])
            print('Keeping {} nyms out of {}'.format(len(keep_ids), len(probes)))
        else:
            num_triplets = len(loaded)
            keep_ids = np.arange(num_triplets)
        return probes[keep_ids].tolist(), syns[keep_ids].tolist(), ants[keep_ids].tolist()

    def make_test_candidates_mat(self, sims, verbose=True):
        if self.nym_type == 'antonym':
            nyms = self.ants
            distractors = self.syns
            neutrals = np.roll(self.ants, 1)
        elif self.nym_type == 'synonym':
            nyms = self.syns
            distractors = self.ants
            neutrals = np.roll(self.syns, 1)
        else:
            raise AttributeError('Invalid arg to "nym_type".')
        first_neighbors = []
        second_neighbors =[]
        for n, p in enumerate(self.probes):  # otherwise argmax would return index for the probe itself
            ids = np.argsort(sims[n])[::-1][:10]
            nearest_neighbors = [self.col_words[i] for i in ids]
            first_neighbors.append(nearest_neighbors[1])  # don't use id=zero because it returns the probe
            second_neighbors.append(nearest_neighbors[2])
        #
        res = []
        for i in range(self.num_probes):
            if first_neighbors[i] == nyms[i]:
                first_neighbors[i] = neutrals[i]
            if second_neighbors[i] == nyms[i]:
                second_neighbors[i] = neutrals[i]
            candidates = [nyms[i], distractors[i], first_neighbors[i], second_neighbors[i], neutrals[i]]
            if verbose:
                print(self.probes[i])
                print(candidates)
            res.append(candidates)
        res = np.vstack(res)
        return res

    def make_data(self, w2e, fold_id, shuffled):
        # train/test split
        x1_train = []
        x2_train = []
        y_train = []
        x1_test = []
        x2_test = []
        for n, (probes, candidate_rows) in enumerate(zip(
                np.array_split(self.probes, config.NymMatching.num_folds),
                np.array_split(self.test_candidates_mat, config.NymMatching.num_folds))):
            nyms, distractors, first_neighbors, second_neighbors, neutrals = candidate_rows.T
            if n != fold_id:
                x1_train += [w2e[p] for p in probes] + [w2e[p] for p in probes]
                x2_train += [w2e[n] for n in nyms] + [w2e[d] for d in distractors]
                y_train += [1] * len(probes) + [0] * len(probes)
                if config.NymMatching.train_on_second_neighbors:
                    x1_train += [w2e[p] for p in probes] + [w2e[p] for p in probes]
                    x2_train += [w2e[n] for n in nyms] + [w2e[d] for d in second_neighbors]
                    y_train += [1] * len(probes) + [0] * len(probes)
            else:
                # assume first one is always correct nym
                dim1 = candidate_rows.shape[1]
                x1_test += [[w2e[p]] * dim1 for p in probes]
                x2_test += [[w2e[nym], w2e[d], w2e[fn], w2e[sn], w2e[n]] for nym, d, fn, sn, n in zip(
                        nyms, distractors, first_neighbors, second_neighbors, neutrals)]
        x1_train = np.vstack(x1_train)
        x2_train = np.vstack(x2_train)
        y_train = np.array(y_train)
        x1_test = np.array(x1_test)
        x2_test = np.array(x2_test)
        # shuffle x-y mapping
        if shuffled:
            print('Shuffling probe-nym mapping')
            np.random.shuffle(y_train)
        return x1_train, x2_train, y_train, x1_test, x2_test

    @staticmethod
    def make_graph(embed_size, shuffled):  # TODO if multiple trials, need to rename scope each time

        def siamese_leg(x, wy):
            y = tf.matmul(x, wy)
            return y

        class Graph:
            with tf.Graph().as_default():
                with tf.device('/{}:0'.format(config.NymMatching.device)):
                    # placeholders
                    x1 = tf.placeholder(tf.float32, shape=(None, embed_size))
                    x2 = tf.placeholder(tf.float32, shape=(None, embed_size))
                    y = tf.placeholder(tf.float32, [None])
                    # siamese
                    with tf.variable_scope('siamese_{}'.format(shuffled), reuse=tf.AUTO_REUSE) as scope:
                        wy = tf.get_variable('wy', shape=[embed_size, config.NymMatching.num_output], dtype=tf.float32)
                        o1 = siamese_leg(x1, wy)
                        o2 = siamese_leg(x2, wy)
                    # loss
                    labels_t = y
                    labels_f = tf.subtract(1.0, y)
                    eucd2 = tf.pow(tf.subtract(o1, o2), 2)
                    eucd2 = tf.reduce_sum(eucd2, 1)
                    eucd = tf.sqrt(eucd2 + 1e-6)
                    C = tf.constant(config.NymMatching.margin)
                    # yi*||o1-o2||^2 + (1-yi)*max(0, C-||o1-o2||^2)
                    pos = tf.multiply(labels_t, eucd2)
                    neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2))
                    losses = tf.add(pos, neg)
                    loss_no_reg = tf.reduce_mean(losses)
                    regularizer = tf.nn.l2_loss(wy)
                    loss = tf.reduce_mean((1 - config.Categorization.beta) * loss_no_reg +
                                          config.NymMatching.beta * regularizer)
                    optimizer = tf.train.AdadeltaOptimizer(learning_rate=config.NymMatching.learning_rate)
                    step = optimizer.minimize(loss)
                # session
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())

        return Graph()

    def train_and_score_expert(self, embedder):
        self.trials = []  # need to flush trials (because multiple embedders)
        bools = [False, True] if config.NymMatching.run_shuffled else [False]
        for shuffled in bools:
            trial = Trial(name='shuffled' if shuffled else '')
            for fold_id in range(config.NymMatching.num_folds):
                # train
                print('Fold {}/{}'.format(fold_id + 1, config.NymMatching.num_folds))
                print('Training nym_matching expert {}...'.format(
                    'with shuffled in-out mapping' if shuffled else ''))
                graph = self.make_graph(embedder.dim1, shuffled)
                data = self.make_data(embedder.w2e, fold_id, shuffled)
                self.train_expert_on_train_fold(graph, trial, data, fold_id)
            self.trials.append(trial)
        # expert_score
        mean_test_acc_traj = self.trials[0].test_acc_trajs.mean(axis=1)
        best_eval_id = np.argmax(mean_test_acc_traj)
        expert_score = mean_test_acc_traj[best_eval_id]
        print('Expert score={:.2f} (at eval step {})'.format(expert_score, best_eval_id))
        return expert_score

    @staticmethod
    def generate_random_train_batches(x1, x2, y, num_probes, num_steps):
        random_choices = np.random.choice(num_probes, config.NymMatching.mb_size * num_steps)
        row_ids_list = np.split(random_choices, num_steps)
        for n, row_ids in enumerate(row_ids_list):
            assert len(row_ids) == config.NymMatching.mb_size
            x1_batch = x1[row_ids]
            x2_batch = x2[row_ids]
            y_batch = y[row_ids]
            yield n, x1_batch, x2_batch, y_batch

    def train_expert_on_train_fold(self, graph, trial, data, fold_id):
        x1_train, x2_train, y_train, x1_test, x2_test = data
        num_train_probes, num_test_probes = len(x1_train), len(x1_test)
        num_train_steps = num_train_probes // config.NymMatching.mb_size * config.NymMatching.num_epochs
        eval_interval = num_train_steps // config.NymMatching.num_evals
        eval_steps = np.arange(0, num_train_steps + eval_interval,
                               eval_interval)[:config.NymMatching.num_evals].tolist()  # equal sized intervals
        print('Train data size: {:,} | Test data size: {:,}'.format(num_train_probes, num_test_probes))
        # training and eval
        for step, x1_batch, x2_batch, y_batch in self.generate_random_train_batches(x1_train, x2_train, y_train,
                                                                                    num_train_probes, num_train_steps):
            if step in eval_steps:
                eval_id = eval_steps.index(step)
                # train loss - can't evaluate accuracy because training requires 1:1 match vs. non-match
                train_loss = graph.sess.run(graph.loss, feed_dict={graph.x1: x1_train,
                                                                   graph.x2: x2_train,
                                                                   graph.y: y_train})

                # test acc - use multiple x2 (unlike training)
                num_correct_test = 0
                for x1_mat, x2_mat in zip(x1_test, x2_test):
                    eucd = graph.sess.run(graph.eucd, feed_dict={graph.x1: x1_mat,
                                                                 graph.x2: x2_mat})
                    if np.argmin(eucd) == 0:  # first one is always nym
                        num_correct_test += 1
                test_acc = num_correct_test / num_test_probes
                trial.test_acc_trajs[eval_id, fold_id] = test_acc
                print('step {:>6,}/{:>6,} |Train Loss={:>7.2f} |Test Acc={:.2f} p={:.4f}'.format(
                    step,
                    num_train_steps - 1,
                    train_loss, test_acc,
                    self.pval(test_acc, n=num_test_probes)))
            # train
            graph.sess.run([graph.step], feed_dict={graph.x1: x1_batch, graph.x2: x2_batch, graph.y: y_batch})

    def save_figs(self, embedder):
        for trial in self.trials:
            # figs
            for fig, fig_name in make_nym_figs():
                p = config.Dirs.runs / embedder.time_of_init / self.name / '{}_{}.png'.format(fig_name, trial.name)
                if not p.parent.exists():
                    p.parent.mkdir(parents=True)
                fig.savefig(str(p))
                print('Saved {} to {}'.format(fig_name, p))

    @property
    def chance(self):
        res = 1 / self.test_candidates_mat.shape[1]
        return res

    def pval(self, prop, n):
        """
        1-tailed: is observed proportion higher than chance?
        pval=1.0 when prop=0 because it is almost always true that a prop > 0.0 is observed
        assumes that proportion of correct responses is binomially distributed
        """
        pval_binom = 1 - binom.cdf(k=prop * n, n=n, p=self.chance)
        return pval_binom

    def score_novice(self, sims, verbose=False):
        self.test_candidates_mat = self.make_test_candidates_mat(sims)  # constructed here because it requires sims
        num_correct = 0
        for n, test_candidates in enumerate(self.test_candidates_mat):
            candidate_sims = sims[n, [self.col_words.index(w) for w in test_candidates]]
            if verbose:
                print(np.around(candidate_sims, 2))
            if np.all(candidate_sims[1:] < candidate_sims[0]):  # correct is always in first position
                num_correct += 1
        result = num_correct / self.num_probes
        print('Novice Accuracy={:.2f} (chance={:.2f}, p={:.4f})'.format(
            result,
            self.chance,
            self.pval(result, n=self.num_probes)))
        return result
