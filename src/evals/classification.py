import numpy as np
import tensorflow as tf
from scipy.stats import binom
from itertools import product

from src import config
from src.figs import make_cat_label_detection_figs
from src.evals.base import EvalBase


class Params:
    beta = [0.0, 0.3]
    num_epochs = [500]
    mb_size = [8]
    learning_rate = [0.1]
    num_hiddens = [32, 256]
    shuffled = [False, True]  # TODO params are graph -specific - put in graph module


class Classification(EvalBase):
    def __init__(self, data_name1, data_name2=None):
        if data_name2 is not None:
            name = '{}_{}_matching'.format(data_name1, data_name2)  # reversed is okay
        else:
            name = '{}_matching'.format(data_name1)
        super().__init__(name, Params)
        #
        self.data_name1 = data_name1
        self.data_name2 = data_name2
        #
        self.probes, self.probe_relata = self.load_probes()
        self.probe2relata = {p: r for p, r in zip(self.probes, self.probe_relata)}
        self.relata = sorted(np.unique(np.concatenate(self.probe_relata)).tolist())
        self.num_probes = len(self.probes)
        self.num_relata = len(self.relata)
        # sims
        self.row_words = self.probes
        self.col_words = self.relata
        # misc
        self.novice_probe_results = None  # TODO test figures

    # ///////////////////////////////////////////// Overwritten Methods START

    def init_results_data(self, trial):  # TODO how to preserve classification figs when classification is just an arhcitecture?
        """
        add task-specific attributes to EvalData class implemented in TaskBase
        """
        res = super().init_results_data(trial)
        res.train_acc_trajs = np.zeros((config.Eval.num_evals,
                                        config.Eval.num_folds))
        res.test_acc_trajs = np.zeros((config.Eval.num_evals,
                                       config.Eval.num_folds))
        res.train_softmax_probs = np.zeros((self.num_probes,
                                            config.Eval.num_evals,
                                            config.Eval.num_folds))
        res.test_softmax_probs = np.zeros((self.num_probes,
                                           config.Eval.num_evals,
                                           config.Eval.num_folds))
        res.trained_test_softmax_probs = np.zeros((self.num_probes,
                                                   config.Eval.num_evals,
                                                   config.Eval.num_folds))
        res.x_mat = np.zeros((self.num_relata,
                              config.Eval.num_evals,
                              config.Eval.num_folds))
        res.cms = []  # confusion matrix (1 per fold)
        return res

    def make_eval_data(self, verbose=False):

        # TODO split architecture from eval

        raise NotImplementedError('classification is an architecture, not an evaluation')

    def split_and_vectorize_eval_data(self, trial, w2e, fold_id):  # TODO test
        # split
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        train_probes = []
        test_probes = []
        for probe, relata in self.probe2relata.items():
            for n, relata_chunk in enumerate(np.array_split(relata, config.Eval.num_folds)):
                probes = [probe] * len(relata_chunk)
                xs = [w2e[p] for p in probes]
                ys = [self.relata.index(relatum) for relatum in relata_chunk]
                if n != fold_id:
                    x_train += xs
                    y_train += ys
                    train_probes += probes
                else:
                    x_test += xs
                    y_test += ys
                    test_probes += probes
        x_train = np.vstack(x_train)
        y_train = np.vstack(y_train)
        x_test = np.vstack(x_test)
        y_test = np.vstack(y_test)
        # shuffle x-y mapping
        if trial.params.shuffled:
            print('Shuffling in-out mapping')
            np.random.shuffle(y_train)
        return x_train, np.squeeze(y_train), x_test, np.squeeze(y_test), train_probes, test_probes

    def make_graph(self, trial, embed_size):  # TODO need multilabel graph - not softmax?
        class Graph:
            with tf.Graph().as_default():
                with tf.device('/{}:0'.format(config.Eval.device)):
                    # placeholders
                    x = tf.placeholder(tf.float32, shape=(None, embed_size))
                    y = tf.placeholder(tf.int32, shape=None)
                    with tf.name_scope('hidden'):
                        wx = tf.get_variable('wx', shape=[embed_size, trial.params.num_hiddens],
                                             dtype=tf.float32)
                        bx = tf.Variable(tf.zeros([trial.params.num_hiddens]))
                        hidden = tf.nn.tanh(tf.matmul(x, wx) + bx)
                    with tf.name_scope('logits'):
                        if trial.params.num_hiddens > 0:
                            wy = tf.get_variable('wy', shape=(trial.params.num_hiddens, self.num_relata),
                                                 dtype=tf.float32)
                            by = tf.Variable(tf.zeros([self.num_relata]))
                            logits = tf.matmul(hidden, wy) + by
                        else:
                            wy = tf.get_variable('wy', shape=[embed_size, self.num_relata],
                                                 dtype=tf.float32)
                            by = tf.Variable(tf.zeros([self.num_relata]))
                            logits = tf.matmul(x, wy) + by
                    softmax = tf.nn.softmax(logits)
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
                    loss_no_reg = tf.reduce_mean(cross_entropy, name='xentropy_mean')
                    regularizer = tf.nn.l2_loss(wx) + tf.nn.l2_loss(wy)
                    loss = tf.reduce_mean((1 - trial.params.beta) * loss_no_reg +
                                          trial.params.beta * regularizer)
                    optimizer = tf.train.AdadeltaOptimizer(learning_rate=trial.params.learning_rate)
                    step = optimizer.minimize(loss)
                with tf.device('/cpu:0'):
                    correct = tf.nn.in_top_k(logits, y, 1)
                    num_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
                # session
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
        return Graph()





