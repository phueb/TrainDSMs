import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization
from sortedcontainers import SortedDict
import pandas as pd

from src import config
from src.figs import make_categorizer_figs
from src.tasks import w2freq


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


class NymSelection:  # SynoNYM & AntoNYM
    def __init__(self, nym_type):
        self.name = '{}selection'.format(nym_type)
        self.nym_type = nym_type
        # evaluation
        self.trials = []  # each result is a class with many attributes

    def make_synonym_task_folds(self, pos='verb'):  # TODO allow this fn to build NOUN syn task


        # make yes_tokens2
        excluded = []
        yes_tokens1 = []
        yes_tokens2 = []
        for n, token1 in enumerate(filtered_token_list):
            synonyms = token_syns_dict[token1]
            filtered_syns = [syn for syn in synonyms
                             if syn in filtered_token_list]
            if filtered_syns:
                token2 = filtered_syns[0]
                yes_tokens1.append(token1)
                yes_tokens2.append(token2)
            else:
                excluded.append(token1)  # excluded due to filtering
        # make yes questions
        yes_lines = []
        for token1, token2 in zip(yes_tokens1, yes_tokens2):
            yes_line = self.task_question.replace('EOI1', token1).replace('EOI2', token2) + ' yes'
            yes_lines.append(yes_line)
        # make no questions
        no_lines = []
        no_tokens2 = None
        while no_tokens2 is None:
            no_tokens2 = self.shuffle_no_overlap(yes_tokens2)  # returns None sometimes
        for token1, token2 in zip(yes_tokens1, no_tokens2):
            no_line = self.task_question.replace('EOI1', token1).replace('EOI2', token2) + ' no'
            no_lines.append(no_line)
        # make folds
        yes_lines, no_lines = self.shuffle_in_unison(yes_lines, no_lines)
        syn_task_folds = self.populate_folds(yes_lines, no_lines, self.mb_size)
        # print
        num_total_lines = len(list(chain(*syn_task_folds)))
        print('Num Excluded terms: {}'.format(
            len(excluded)))
        print('Number of syn_task_lines: {}/{}'.format(
            num_total_lines, len(filtered_token_list) * 2))
        print('Lost {} lines due to mini batching'.format(
            len(filtered_token_list) * 2 - num_total_lines - (len(excluded) * 2)))
        # check
        assert self.is_bal(yes_lines, no_lines, yes_tokens1, self.eoi1_id)
        assert self.is_bal(yes_lines, no_lines, yes_tokens2, self.eoi2_id)
        return syn_task_folds


    def train_and_score_expert(self, w2e, embed_size):
        for shuffled in [False, True]:
            name = 'shuffled' if shuffled else ''
            trial = Trial(name=name,
                          num_cats=self.num_cats,
                          g=self.make_classifier_graph(embed_size),
                          data=self.make_data(w2e, shuffled))
            print('Training categorization expert with {} categories...'.format(name))
            self.train_expert(trial)
            self.trials.append(trial)

        # expert_score
        expert_score = self.trials[0].test_acc_traj[-1]
        return expert_score

    def score_novice(self, probe_simmat, probe_cats=None, metric='ba'):
        result = None
        return result
