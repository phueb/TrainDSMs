import numpy as np
import tensorflow as tf

from src import config
from src.figs import make_nym_figs


class Trial(object):
    def __init__(self, name):
        self.name = name
        self.train_acc_trajs = np.zeros((config.NymMatching.num_evals, config.NymMatching.num_folds))
        self.test_acc_trajs = np.zeros((config.NymMatching.num_evals, config.NymMatching.num_folds))


def siamese_leg(x):
    return x


class NymMatching:
    def __init__(self, pos, nym_type):
        self.name = '{}_{}_matching'.format(pos, nym_type)
        self.pos = pos
        self.nym_type = nym_type
        self.probes, self.nyms = self.load_data()
        self.num_pairs = len(self.probes)
        self.distractor_nym_ids_mat = self.make_distractor_nym_ids_mat()
        # evaluation
        self.trials = None

    @property
    def row_words(self):  # used to build sims
        return self.probes

    @property
    def col_words(self):
        return self.nyms

    def load_data(self):
        p = config.Dirs.tasks / self.name / '{}_{}.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
        print('Loading {}'.format(p))
        both = np.loadtxt(p, dtype='str')
        np.random.shuffle(both)
        probes, nyms = both.T
        # remove duplicate nyms
        if config.NymMatching.remove_duplicate_nyms:
            keep_ids = []
            nym_set = set()
            num_pairs = len(both)
            for i in range(num_pairs):
                if nyms[i] not in nym_set:
                    keep_ids.append(i)
                nym_set.add(nyms[i])
        else:
            num_pairs = len(both)
            keep_ids = np.arange(num_pairs)
        return probes[keep_ids].tolist(), nyms[keep_ids].tolist()

    def make_distractor_nym_ids_mat(self):
        res = []
        for n in range(self.num_pairs):
            choices = [i for i in range(self.num_pairs) if i != n]
            distractor_nym_ids = np.random.choice(choices, config.NymMatching.num_distractors, replace=False)
            res.append(distractor_nym_ids)
        res = np.vstack(res)
        return res

    def make_data(self, w2e, fold_id, shuffled):
        # train/test split (separately for each category)
        x1_train = []
        x2_train = []
        y_train = []
        x1_test = []
        x2_test = []
        y_test = []
        # permute nyms to result in distractors without duplicates and without "overlap" (with nyms)
        distractors = np.roll(self.nyms, 1)
        assert np.all(distractors != self.nyms)
        assert len(set(distractors)) == len(distractors)
        for n, (probes_in_fold, nyms_in_fold, distractors_in_fold) in enumerate(zip(
                np.array_split(self.probes, config.NymMatching.num_folds),
                np.array_split(distractors, config.NymMatching.num_folds),
                np.array_split(self.nyms, config.NymMatching.num_folds))):
            x1s = [w2e[p] for p in probes_in_fold] + [w2e[p] for p in probes_in_fold]
            x2s = [w2e[p] for p in nyms_in_fold] + [w2e[p] for p in distractors_in_fold]
            ys = [1] * len(nyms_in_fold) + [0] * len(nyms_in_fold)
            assert len(x1s) == len(x2s) == len(ys)
            if n != fold_id:
                x1_train += x1s
                x2_train += x2s
                y_train += ys
            else:
                x1_test += x1s
                x2_test += x2s
                y_test += ys
        x1_train = np.vstack(x1_train)
        x2_train = np.vstack(x2_train)
        y_train = np.array(y_train)
        x1_test = np.vstack(x1_test)
        x2_test = np.vstack(x2_test)
        y_test = np.array(y_test)
        # shuffle x-y mapping
        if shuffled:
            print('Shuffling category assignment')
            np.random.shuffle(y_train)
        return x1_train, x2_train, y_train, x1_test, x2_test, y_test

    @staticmethod
    def make_matcher_graph(embed_size):
        class Graph:
            with tf.Graph().as_default():
                with tf.device('/{}:0'.format(config.NymMatching.device)):
                    # placeholders
                    x1 = tf.placeholder(tf.float32, shape=(None, embed_size))
                    x2 = tf.placeholder(tf.float32, shape=(None, embed_size))
                    with tf.variable_scope("siamese") as scope:
                        o1 = siamese_leg(x1)
                        scope.reuse_variables()
                        o2 = siamese_leg(x2)
                    # loss
                    y = tf.placeholder(tf.float32, [None])
                    abs_dist = tf.abs(o1 - o2)
                    logits = tf.layers.dense(abs_dist, 1)
                    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(logits), labels=y)
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.NymMatching.learning_rate)
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
                graph = self.make_matcher_graph(embedder.dim1)
                data = self.make_data(embedder.w2e, fold_id, shuffled)
                self.train_expert_on_train_folds(graph, trial, data, fold_id)
            self.trials.append(trial)
            # expert_score
        expert_score = np.mean(self.trials[0].test_acc_trajs[-1].mean())
        return expert_score

    def score_novice(self, sims, verbose=True):
        """
        sims should be matrix of shape [num probes, num syms] where value at [i, j] is sim between probe i and nym j
        the number of probes is not the number of types - repeated occurrences are allowed and counted
        """
        num_correct = 0
        for n, distractor_nym_ids in enumerate(self.distractor_nym_ids_mat):
            distractor_sims = sims[n, distractor_nym_ids]
            correct_sim = sims[n, n]  # correct pairs are in diagonal
            if np.all(distractor_sims < correct_sim):
                num_correct += 1
            if verbose:
                print('distractor_sims')
                print(distractor_sims)
                print('correct_sim')
                print(correct_sim)
        result = num_correct / self.num_pairs
        print('Accuracy at {} = {:.2f}'.format(self.name, result))
        print('Chance = {:.2f}'.format(1 / (config.NymMatching.num_distractors + 1)))

        return result

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

    def train_expert_on_train_folds(self, graph, trial, data, fold_id):
        x1_train, x2_train, y_train, x1_test, x2_test, y_test = data
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

                # TODO debug
                logits = graph.sess.run(graph.logits, feed_dict={graph.x1: x1_train,
                                                                 graph.x2: x2_train,
                                                                 graph.y: y_train})

                print(logits)
                print(logits.shape)
                print(y_train)
                print(y_train.shape)

                # TODO to evaluate - need to compare all pairs


                trial.test_acc_trajs[eval_id, fold_id] = test_acc
                print('step {:>6,}/{:>6,} |Train Acc={:.2f} |Test Acc={:.2f}'.format(
                    step,
                    num_train_steps - 1,
                    train_acc,
                    test_acc))
            # train
            graph.sess.run([graph.step], feed_dict={graph.x1: x1_batch, graph.x2: x2_batch, graph.y: y_batch})

    def save_figs(self, time_of_init):  # TODO implement
        for trial in self.trials:
            # figs
            for fig, fig_name in make_nym_figs():
                p = config.Dirs.runs / time_of_init / self.name / '{}_{}.png'.format(fig_name, trial.name)
                if not p.parent.exists():
                    p.parent.mkdir(parents=True)
                fig.savefig(str(p))
                print('Saved {} to {}'.format(fig_name, p))
