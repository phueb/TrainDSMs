import numpy as np

from src import config


class Trial(object):  # TODO make this available to all experts?
    def __init__(self, name, num_cats):
        self.name = name
        self.train_acc_trajs_p = np.zeros((config.Categorization.num_evals, config.Categorization.num_folds))
        self.test_acc_trajs_p = np.zeros((config.Categorization.num_evals, config.Categorization.num_folds))
        self.train_acc_trajs_by_cat = np.zeros((num_cats,
                                                config.Categorization.num_evals,
                                                config.Categorization.num_folds))
        self.test_acc_trajs_by_cat = np.zeros((num_cats,
                                               config.Categorization.num_evals,
                                               config.Categorization.num_folds))
        self.x_mat = np.zeros((num_cats,
                               config.Categorization.num_evals,
                               config.Categorization.num_folds))
        self.cms = []  # confusion matrix (1 per fold)


class NymMatching:
    def __init__(self, pos, nym_type):
        self.name = '{}_{}_matching'.format(pos, nym_type)
        self.pos = pos
        self.nym_type = nym_type
        self.probes, self.nyms = self.load_data()
        self.num_pairs = len(self.probes)
        self.distractor_nym_ids_mat = self.make_distractor_ids_mat()
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
        probes, nyms = np.loadtxt(p, dtype='str').T
        return probes.tolist(), nyms.tolist()

    def make_distractor_ids_mat(self):
        res = []
        for n in range(self.num_pairs):
            choices = [i for i in range(self.num_pairs) if i != n]
            res.append(np.random.choice(choices, config.NymMatching.num_distractors, replace=False))
        res = np.vstack(res)
        return res


    def make_data(self, w2e, shuffled):
        # put data in array
        x = np.vstack([w2e[p] for p in self.probes])  # TODO need 5 probes per input with only one correct x
        y = np.vstack([])

        # TODO use same distractors as novice

        # shuffle x-y mapping
        if shuffled:
            print('Shuffling probe-nym mapping')
            np.random.shuffle(y_train)
        return x_train, y_train, x_test, y_test

    def train_and_score_expert(self, embedder):  # TODO implement
        self.trials = []  # need to flush trials (because multiple embedders)
        bools = [False, True] if config.Categorization.run_shuffled else [False]
        for shuffled in bools:
            trial = Trial(name='shuffled' if shuffled else '',
                          num_cats=self.num_cats)
            for fold_id in range(config.Categorization.num_folds):
                # train
                print('Fold {}/{}'.format(fold_id + 1, config.Categorization.num_folds))
                print('Training categorization expert {}...'.format(
                    'with shuffled in-out mapping' if shuffled else ''))
                graph = self.make_classifier_graph(embedder.dim1)
                data = self.make_data(embedder.w2e, embedder.w2freq, fold_id, shuffled)
                self.train_expert_on_fold(graph, trial, data, fold_id)
                # add confusion mat to trial
                x_test = data[2]
                y_test = data[3]
                logits = graph.sess.run(graph.logits,
                                        feed_dict={graph.x: x_test, graph.y: y_test}).astype(np.int)
                y_pred = np.argmax(logits, axis=1).astype(np.int)
                cm = np.zeros((self.num_cats, self.num_cats))
                for yt, yp in zip(y_test, y_pred):
                    cm[yt, yp] += 1
                trial.cms.append(cm)
            self.trials.append(trial)

    def score_novice(self, sims):
        """
        sims should be matrix of shape [num probes, num syms] where value at [i, j] is sim between probe i and nym j
        the number of probes is not the number of types - repeated occurrences are allowed and counted
        """
        num_correct = 0
        for n, distractor_nym_ids in enumerate(self.distractor_nym_ids_mat):

            candidate_sims = sims[n, distractor_nym_ids]  # TODO num_distractors or num_candidates ?
            correct_sim = sims[n, n]  # correct pairs are in diagonal
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
            cm = (cm / cat_freqs_mat).astype(np.int)
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

