import numpy as np
from sortedcontainers import SortedDict

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
        self.p2nyms = self.make_p2nym()
        # evaluation
        self.trials = []  # each result is a class with many attributes

    def make_p2nym(self):
        res = SortedDict()
        p = config.Global.task_dir / self.name / '{}_{}.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
        with p.open('r') as f:
            for line in f:
                data = line.strip().strip('\n').split()
                probe = data[0]
                nym = data[1]  # # TODO multiple probes may occur with different nyms ?
                res[probe] = nym
                print(probe, nym)
        return res

    def make_data(self, w2e, shuffled):
        # put data in array
        x = np.vstack([w2e[p] for p in self.probes])
        y = np.zeros(x.shape[0], dtype=np.int)
        for n, probe in enumerate(self.probes):
            cat = self.p2cat[probe]
            cat_id = self.cats.index(cat)
            y[n] = cat_id
        # split
        max_f = config.Categorization.max_freq
        w2freq = make_w2freq(config.Corpus.name)
        probe_freq_list = [w2freq[probe] if w2freq[probe] < max_f else max_f for probe in self.probes]
        test_ids = np.random.choice(self.num_probes,
                                    size=int(self.num_probes * config.Categorization.test_size),
                                    replace=False)
        train_ids = [i for i in range(self.num_probes) if i not in test_ids]
        assert len(train_ids) + len(test_ids) == self.num_probes
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
            print('Shuffling category assignment')
            np.random.shuffle(y_train)
        return x_train, y_train, x_test, y_test

    def train_and_score_expert(self, w2e, embed_size):
        for shuffled in [False, True]:
            name = 'shuffled' if shuffled else ''
            trial = Trial(name=name,
                          num_cats=self.num_cats,
                          g=self.make_classifier_graph(embed_size),
                          data=self.make_data(w2e, shuffled))
            print('Training semantic_categorization expert with {} categories...'.format(name))
            self.train_expert(trial)
            self.trials.append(trial)

        # expert_score
        expert_score = self.trials[0].test_acc_traj[-1]
        return expert_score

    def score_novice(self, probe_simmat, probe_cats=None, metric='ba'):
        result = None
        return result
