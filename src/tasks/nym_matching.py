import numpy as np

from src import config


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


class NymMatching:
    def __init__(self, pos, nym_type):
        self.name = '{}_{}_matching'.format(pos, nym_type)
        self.pos = pos
        self.nym_type = nym_type
        self.probes, self.nyms = self.load_data()
        self.num_pairs = len(self.probes)
        self.correct_nym_ids = self.make_correct_candidate_ids()
        self.distractor_nym_ids_mat = self.make_distractor_ids_mat()
        # evaluation
        self.trials = []  # each result is a class with many attributes

    @property
    def row_words(self):  # used to build sims
        return self.probes

    @property
    def col_words(self):
        return self.nyms

    def load_data(self):
        p = config.Dirs.tasks / self.name / '{}_{}.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
        print('Loading {}'.format(p))
        probes, nyms = np.loadtxt(p, dtype='str').T
        return probes.tolist(), nyms.tolist()

    def make_correct_candidate_ids(self):
        """
        return list of integers with length [num pairs] where each integer is a candidate_id
        specifying which candidate in a given trial is the correct nym
        """
        res = np.random.choice(config.NymMatching.num_distractors + 1, size=self.num_pairs)
        return res

    def make_distractor_ids_mat(self):  # TODO test
        res = np.random.choice(self.num_pairs, (self.num_pairs, config.NymMatching.num_distractors))
        return res


    def make_data(self, w2e, shuffled):
        # put data in array
        x = np.vstack([w2e[p] for p in self.probes])  # TODO need 5 probes per input with only one correct x
        y = np.vstack([])


        # TODO use make_test_word_ids_mat

        for n, probe in enumerate(self.probes):
            cat = self.p2cat[probe]
            cat_id = self.cats.index(cat)
            y[n] = cat_id
        # split
        max_f = config.Categorization.max_freq
        w2freq = make_w2freq(config.Corpus.name)
        probe_freq_list = [w2freq[probe] if w2freq[probe] < max_f else max_f for probe in self.probes]
        test_ids = np.random.choice(self.num_pairs,
                                    size=int(self.num_pairs * config.Categorization.test_size),
                                    replace=False)
        train_ids = [i for i in range(self.num_pairs) if i not in test_ids]
        assert len(train_ids) + len(test_ids) == self.num_pairs
        x_train = x[train_ids, :]
        y_train = y[train_ids]
        pfs_train = np.array(probe_freq_list)[train_ids]
        x_test = x[test_ids, :]
        y_test = y[test_ids]
        # repeat train samples proportional to corpus frequency - must happen AFTER split
        x_train = np.repeat(x_train, pfs_train, axis=0)  # repeat according to token frequency
        y_train = np.repeat(y_train, pfs_train, axis=0)
        # shuffle x-y mapping
        if shuffled:
            print('Shuffling probe-nym mapping')
            np.random.shuffle(y_train)
        return x_train, y_train, x_test, y_test

    def train_and_score_expert(self, w2e, embed_size):  # TODO implement
        for shuffled in [False, True]:
            name = 'shuffled' if shuffled else ''
            trial = Trial(name=name,
                          num_cats=self.num_cats,
                          g=self.make_matcher_graph(embed_size),
                          data=self.make_data(w2e, shuffled))
            print('Training semantic_categorization expert with {} categories...'.format(name))
            self.train_expert(trial)
            self.trials.append(trial)

        # expert_score
        expert_score = self.trials[0].test_acc_traj[-1]
        return expert_score

    def score_novice(self, sims):
        """
        sims should be matrix of shape [num probes, num syms] where value at [i, j] is sim between probe i and nym j
        the number of probes is not the number of types - repeated occurrences are allowed and counted
        """
        num_correct = 0
        for n, (distractor_nym_ids, correct_nym_id) in enumerate(zip(self.distractor_nym_ids_mat, self.correct_nym_ids)):

            candidate_sims = sims[n, distractor_nym_ids]
            correct_sim = sims[n, correct_nym_id]
            if np.all(candidate_sims < correct_sim):
                num_correct += 1

            # print(candidate_sims)
            # print(correct_sim)

        result = num_correct / self.num_pairs

        print('Accuracy at {} = {:.2f}'.format(self.name, result))
        print('Chance = {:.2f}'.format(1 / config.NymMatching.num_distractors))

        return result

    def save_figs(self, embedder_name):  # TODO implement
        for trial in self.trials:
            x_train, y_train, x_test, y_test = trial.data
            dummies = pd.get_dummies(y_test)
            cat_freqs = np.sum(dummies.values, axis=0)
            num_test_cats = len(set(y_test))
            cat_freqs_mat = np.tile(cat_freqs, (num_test_cats, 1))  # not all categories may be in test data
            # confusion mat
            logits = trial.g.sess.run(trial.g.logits, feed_dict={trial.g.x: x_test, trial.g.y: y_test}).astype(np.int)
            predicted_y = np.argmax(logits, axis=1).astype(np.int)
            cm = np.zeros((num_test_cats, num_test_cats))
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

