import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cytoolz import itertoolz

from src import config


def make_nym_figs():
    return []


def make_cat_member_verification_figs():
    return []


def make_categorizer_figs(feature_diagnosticity_mat,
                          train_acc_traj,
                          test_acc_traj,
                          train_softmax_traj,
                          test_softmax_traj,
                          cm,
                          cum_x_mat,
                          novice_results_by_cat,
                          expert_results_by_cat,
                          novice_results_by_probe,
                          expert_results_by_probe,
                          cat2train_evals_to_criterion,
                          cat2test_evals_to_criterion,
                          cat2trained_test_evals_to_criterion,
                          cats):
    num_cats = len(cats)

    def make_feature_diagnosticity_distribution_fig():
        chunk_size = 10  # number of unique default colors
        chunk_ids_list = list(itertoolz.partition_all(chunk_size, np.arange(num_cats)))
        num_rows = len(chunk_ids_list)
        fig, axarr = plt.subplots(nrows=num_rows, ncols=1,
                                  figsize=(config.Figs.width, 4 * num_rows),
                                  dpi=config.Figs.dpi)
        plt.suptitle('How diagnostic are embedding features\nabout category membership?')
        if not isinstance(axarr, np.ndarray):
            axarr = [axarr]
        for ax, chunk_ids in zip(axarr, chunk_ids_list):
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='both', top=False, right=False)
            ax.set_xlabel('F1-score')
            ax.set_ylabel('Frequency')
            ax.set_xlim([0, 1])
            # plot histograms
            for cat, row in zip([cats[i] for i in chunk_ids], feature_diagnosticity_mat[chunk_ids, :]):
                print('Highest f1-score for "{}" is {:.2f}'.format(cat, np.max(row)))
                ax.hist(row,
                        bins=None,
                        linewidth=2,
                        histtype='step',
                        label=cat)
            ax.legend(loc='best')
        plt.tight_layout()
        return fig

    def make_feature_diagnosticity_fig():
        fig, ax = plt.subplots(1, figsize=(config.Figs.width, config.Figs.width), dpi=config.Figs.dpi)
        plt.title('How diagnostic is an embedding feature about a category?')
        sns.heatmap(feature_diagnosticity_mat,
                    ax=ax, square=False, annot=False, cbar_kws={"shrink": .5}, cmap='jet', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks(np.arange(num_cats) + 0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels(cats, rotation=0)
        ax.set_xlabel('Embedding Features')
        # colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['0.0', '0.5', '1.0'])
        cbar.set_label('F1 score')
        plt.tight_layout()
        return fig

    def make_novice_vs_expert_fig(xs, ys, annotations=None):
        """
        Returns fig showing scatterplot of novice vs. expert accuracy
        """
        fig, ax = plt.subplots(1, figsize=(config.Figs.width, config.Figs.width), dpi=config.Figs.dpi)
        ax.set_ylim([0, 1.1])
        ax.set_xlim([0.5, 1.1])
        ax.set_xlabel('Novice Balanced Accuracy', fontsize=config.Figs.axlabel_fontsize)
        ax.set_ylabel('Expert Correct Category Probability', fontsize=config.Figs.axlabel_fontsize)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False)
        # plot
        if annotations:
            it = iter(annotations)
        for x, y in zip(xs, ys):
            ax.scatter(x, y, color='black')
            if annotations:
                ax.annotate(next(it), (x + 0.01, y))
        plt.tight_layout()
        return fig

    def make_traj_fig(train_traj, test_traj):
        """
        Returns fig showing accuracy or correct softmax prob of train and test
        """
        fig, ax = plt.subplots(1, figsize=(config.Figs.width, 3), dpi=config.Figs.dpi)
        ax.set_ylim([0, 1])
        ax.set_xlabel('Number of Training Samples', fontsize=config.Figs.axlabel_fontsize)
        ylabel = 'Accuracy'
        ax.set_ylabel(ylabel, fontsize=config.Figs.axlabel_fontsize)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False)
        # plot
        x = cum_x_mat.sum(axis=0)
        ax.plot(x, train_traj, '-', linewidth=config.Figs.line_width, label='train')
        ax.plot(x, test_traj, '-', linewidth=config.Figs.line_width, label='test')
        ax.legend(loc='best')
        plt.tight_layout()
        return fig

    def make_num_steps_to_criterion_fig():
        """
        Returns fig showing number of evaluation steps needed to achieve 100% accuracy by category for train and test
        """
        fig, ax = plt.subplots(1, figsize=(config.Figs.width, config.Figs.width), dpi=config.Figs.dpi)
        # ax.set_xlim([0, config.Categorization.num_evals])
        ax.set_xlabel('Number of Evaluation Steps to Criterion={}'.format(
            config.Categorization.softmax_criterion),
            fontsize=config.Figs.axlabel_fontsize)
        ylabel = 'Frequency'
        ax.set_ylabel(ylabel, fontsize=config.Figs.axlabel_fontsize)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False)
        # plot
        train_evals_to_criterion = np.concatenate(list(cat2train_evals_to_criterion.values()))
        test_evals_to_criterion = np.concatenate(list(cat2test_evals_to_criterion.values()))
        trained_test_evals_to_criterion = np.concatenate(list(cat2trained_test_evals_to_criterion.values()))
        # trained
        ax.hist(train_evals_to_criterion,
                config.Categorization.num_bins,
                linewidth=3,
                histtype='step',
                label='train')
        # test
        ax.hist(test_evals_to_criterion,
                config.Categorization.num_bins,
                linewidth=3,
                histtype='step',
                label='test')
        # trained test
        ax.hist(trained_test_evals_to_criterion,
                config.Categorization.num_bins,
                linewidth=3,
                histtype='step',
                label='post-train test')
        ax.legend(loc='upper left')
        plt.tight_layout()
        return fig

    def make_num_steps_to_criterion_by_cat_fig():
        """
        Returns fig showing number of evaluation steps needed to achieve 100% accuracy by category for train and test
        """
        fig, axarr = plt.subplots(num_cats, 1,
                                  figsize=(config.Figs.width, 3 * num_cats),
                                  dpi=config.Figs.dpi)
        for ax, cat in zip(axarr, cats):
            ax.set_title(cat)
            ax.set_xlabel('Number of Evaluation Steps to Criterion={}'.format(
                config.Categorization.softmax_criterion),
                fontsize=config.Figs.axlabel_fontsize)
            ylabel = 'Frequency'
            ax.set_ylabel(ylabel, fontsize=config.Figs.axlabel_fontsize)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='both', top=False, right=False)
            ax.yaxis.grid(True)
            # plot
            train_evals_to_criterion = cat2train_evals_to_criterion[cat]
            test_evals_to_criterion = cat2test_evals_to_criterion[cat]
            trained_test_evals_to_criterion = cat2trained_test_evals_to_criterion[cat]
            # train
            ax.hist(train_evals_to_criterion,
                    config.Categorization.num_bins,
                    histtype='step',
                    label='train')
            # test
            ax.hist(test_evals_to_criterion,
                    config.Categorization.num_bins,
                    histtype='step',
                    label='test')
            # trained test
            ax.hist(trained_test_evals_to_criterion,
                    config.Categorization.num_bins,
                    histtype='step',
                    label='post-train test')
            ax.legend(loc='best')
        plt.tight_layout()
        return fig

    def make_cm_fig():
        """
        Returns fig showing confusion matrix
        """
        fig, ax = plt.subplots(1, 1, figsize=(config.Figs.width, config.Figs.width),
                               dpi=config.Figs.dpi)
        # plot
        # sns.heatmap(cm, ax=ax, square=True, annot=False, cbar_kws={"shrink": .5}, cmap='jet', vmin=0, vmax=1)
        sns.heatmap(cm, ax=ax, square=True, annot=False, cbar_kws={"shrink": .5}, cmap='jet')
        # axis 2 (needs to be below plot for axes to be labeled)
        ax.set_yticklabels([cat + ' (truth)' for cat in cats], rotation=0)
        ax.set_xticklabels(cats, rotation=90)
        title = 'Confusion Matrix'
        ax.set_title(title)
        # colorbar
        cbar = ax.collections[0].colorbar
        # cbar.set_ticks([0, 0.5, 1])
        # cbar.set_ticklabels(['0.0', '0.5', '1.0'])
        cbar.set_label('Hits & False Alarms')
        plt.tight_layout()
        return fig

    return [
        (make_feature_diagnosticity_distribution_fig(), 'feature_diagnosticity_distr'),
        (make_feature_diagnosticity_fig(), 'feature_diagnosticity'),
        (make_novice_vs_expert_fig(novice_results_by_cat, expert_results_by_cat, cats), 'nov_vs_exp_cat'),
        (make_novice_vs_expert_fig(novice_results_by_probe, expert_results_by_probe), 'nov_vs_exp_probe'),
        (make_traj_fig(train_acc_traj, test_acc_traj), 'acc_traj'),
        (make_traj_fig(train_softmax_traj, test_softmax_traj,), 'softmax_traj'),
        (make_num_steps_to_criterion_fig(), 'criterion'),
        (make_num_steps_to_criterion_by_cat_fig(), 'criterion_by_cat'),
        (make_cm_fig(), 'cm')
    ]
