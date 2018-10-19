import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import config


def make_categorizer_figs(train_acc_traj,
                          test_acc_traj,
                          train_softmax_traj,
                          test_softmax_traj,
                          cm,
                          x_mat,
                          novice_results_by_cat,
                          expert_results_by_cat,
                          novice_results_by_probe,
                          expert_results_by_probe,
                          cat2train_evals_to_criterion,
                          cat2test_evals_to_criterion,
                          cats):
    num_cats = len(cats)
    max_x = np.max(x_mat[:, -1])

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
        # ax.yaxis.grid(True)
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
        ax.yaxis.grid(True)
        # plot
        x = x_mat.sum(axis=0)
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
        ax.yaxis.grid(True)
        # plot
        train_evals_to_criterion = np.concatenate(list(cat2train_evals_to_criterion.values()))
        test_evals_to_criterion = np.concatenate(list(cat2test_evals_to_criterion.values()))
        # assume that all probes are assigned a value
        # assign max value if a probe doesn't reach criterion before last eval
        assert len(train_evals_to_criterion) == len(test_evals_to_criterion)
        ax.hist(train_evals_to_criterion,
                config.Categorization.num_bins,
                histtype='step',
                label='train')
        ax.hist(test_evals_to_criterion,
                config.Categorization.num_bins,
                histtype='step',
                label='test')
        ax.legend(loc='best')
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
            ax.set_ylim([0, 1])  # y is density
            ax.set_xlim([0, config.Categorization.num_evals])
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
            ax.hist(train_evals_to_criterion,
                    config.Categorization.num_bins,
                    histtype='step',
                    density=1,
                    label='train')
            ax.hist(test_evals_to_criterion,
                    config.Categorization.num_bins,
                    histtype='step',
                    density=1,
                    label='test')
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

    return [(make_novice_vs_expert_fig(novice_results_by_cat, expert_results_by_cat, cats), 'nov_vs_exp_cat'),
            (make_novice_vs_expert_fig(novice_results_by_probe, expert_results_by_probe), 'nov_vs_exp_probe'),
            (make_traj_fig(train_acc_traj, test_acc_traj), 'acc_traj'),
            (make_traj_fig(train_softmax_traj, test_softmax_traj,), 'softmax_traj'),
            (make_num_steps_to_criterion_fig(), 'criterion'),
            (make_cm_fig(), 'cm')]
