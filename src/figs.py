import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import config


def make_categorizer_figs(cm,
                          x_mat,
                          train_acc_traj_by_cat,
                          test_acc_traj_by_cat,
                          novice_results_by_cat,
                          expert_results_by_cat,
                          novice_results_by_probe,
                          expert_results_by_probe,
                          cats):
    train_acc_traj = train_acc_traj_by_cat.mean(axis=0)
    test_acc_traj = test_acc_traj_by_cat.mean(axis=0)
    num_cats = len(cats)
    max_x = np.max(x_mat[:, -1])

    def make_novice_vs_expert_fig(xs, ys, annotations=None):
        """
        Returns fig showing scatterplot of novice vs. expert accuracy by category
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

    def make_acc_traj_fig():
        """
        Returns fig showing accuracy of a classifier tasked to map hidden representations of probes to categories
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
        ax.plot(x, train_acc_traj, '-', linewidth=config.Figs.line_width, label='train')
        ax.plot(x, test_acc_traj, '-', linewidth=config.Figs.line_width, label='test')
        ax.legend(loc='best')
        plt.tight_layout()
        return fig

    def make_acc_by_cat_fig():
        """
        Returns fig showing accuracy broken down by category of a classifier tasked to map hidden representations
         of probes to categories
        """
        fig, axarr = plt.subplots(num_cats, 1,
                                  figsize=(config.Figs.width, 3 * num_cats),
                                  dpi=config.Figs.dpi)
        for ax, train_hca_cat_traj, test_hca_cat_traj, cat, x in zip(
                axarr, train_acc_traj_by_cat, test_acc_traj_by_cat, cats, x_mat):
            ax.set_ylim([0, 1])
            ax.set_xlim([0, max_x])
            ax.set_xlabel('Number of Training Samples', fontsize=config.Figs.axlabel_fontsize)
            ylabel = 'Accuracy'
            ax.set_ylabel(ylabel, fontsize=config.Figs.axlabel_fontsize)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='both', top=False, right=False)
            ax.yaxis.grid(True)
            # plot
            ax.plot(x, train_hca_cat_traj, '-', linewidth=config.Figs.line_width, label='{} (train)'.format(cat))
            ax.plot(x, test_hca_cat_traj, '-', linewidth=config.Figs.line_width, label='{} (test)'.format(cat))
            ax.legend()
        plt.tight_layout()
        return fig

    def make_cm_fig():
        """
        Returns fig showing confusion matrix of classifier trained to map hidden representations of probes to categories
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
            (make_acc_traj_fig(), 'acc'),
            (make_acc_by_cat_fig(), 'acc_by_cat'),
            (make_cm_fig(), 'cm')]
