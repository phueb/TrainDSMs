import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import sem, t
from typing import List, Tuple, Dict, Optional

from traindsms import config


def make_bar_plot(label2accuracies: Dict[str, List[float]],
                  figsize: Tuple[int, int] = (8, 4),
                  title: str = '',
                  xlabel: str = 'Model',
                  ylabel: str = 'Accuracy',
                  width: float = 0.2,
                  confidence: float = 0.95,
                  y_grid: bool = False,
                  ylims: Optional[List[float]] = None,
                  h_line: Optional[float] = None,
                  x_label_threshold: int = 10,
                  ):
    """
    plot average accuracy by group (job) at end of training.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=config.Figs.dpi)
    plt.title(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    if y_grid:
        ax.yaxis.grid(True)
    if ylims is not None:
        ax.set_ylim(ylims)
    else:
        ax.set_ylim([0.0, 1.0])

    num_groups = len(label2accuracies)
    edges = [width * i for i in range(num_groups)]  # x coordinate for each bar-center
    x = np.arange(1)

    # x-axis
    ax.set_xlabel(xlabel, fontsize=config.Figs.ax_font_size)
    ax.set_xticks(edges)
    if len(label2accuracies) > x_label_threshold:
        x_tick_labels = [gn if n == 0 or n == len(label2accuracies) -1 else ''
                         for n, gn in enumerate(label2accuracies)]
    else:
        x_tick_labels = label2accuracies
    ax.set_xticklabels(x_tick_labels, fontsize=6)

    # y axis
    ax.set_ylabel(ylabel, fontsize=config.Figs.ax_font_size)

    # colors
    palette = np.asarray(sns.color_palette('hls', num_groups))
    colors = iter(palette)

    if h_line is not None:
        ax.axhline(y=h_line, color='grey', linestyle=':', zorder=3)

    # plot
    for edge, color, group_name in zip(edges, colors, label2accuracies):
        accuracies = label2accuracies[group_name]
        y = np.mean(accuracies)

        # margin of error (across paradigms, not replications)
        n = len(accuracies)
        h = sem(accuracies, axis=0) * t.ppf((1 + confidence) / 2, n - 1)  # margin of error

        # plot all bars belonging to a single model group (same color)
        ax.bar(x + edge,
               y,
               width,
               yerr=h,
               color=color,
               zorder=3,
               )

    plt.tight_layout()
    return fig


def make_line_plot(label2accuracy_mat: Dict[str, np.array],  # [num groups, num epochs]
                   figsize: Tuple[int, int] = (8, 4),
                   title: str = '',
                   xlabel: str = 'Epoch',
                   ylabel: str = 'Accuracy',
                   confidence: float = 0.95,
                   y_grid: bool = False,
                   ylims: Optional[List[float]] = None,
                   h_line: Optional[float] = None,
                   shrink_xtick_labels: bool = False,
                   ):
    """
    plot average accuracy by group across training.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=config.Figs.dpi)
    plt.title(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    if y_grid:
        ax.yaxis.grid(True)
    if ylims is not None:
        ax.set_ylim(ylims)
    else:
        ax.set_ylim([0.0, 1.0])

    # axes
    ax.set_xlabel(xlabel, fontsize=config.Figs.ax_font_size)
    ax.set_ylabel(ylabel, fontsize=config.Figs.ax_font_size)

    # colors
    num_groups = len(label2accuracy_mat)
    palette = np.asarray(sns.color_palette('hls', num_groups))
    colors = iter(palette)

    if h_line is not None:
        ax.axhline(y=h_line, color='grey', linestyle=':', zorder=1)

    # plot
    max_num_epochs = 1
    for color, group_name in zip(colors, label2accuracy_mat):
        mat = label2accuracy_mat[group_name]
        y_mean = np.mean(mat, axis=0)
        x = np.arange(len(y_mean)) + 1  # + 1 because data is not saved for epoch=0

        # margin of error (across paradigms, not replications)
        n = len(mat)
        h = sem(mat, axis=0) * t.ppf((1 + confidence) / 2, n - 1)  # margin of error

        # plot
        ax.plot(x, y_mean, '-',
                color=color,
                label=group_name,
                )
        ax.fill_between(x,
                        y_mean + h,
                        y_mean - h,
                        alpha=0.2,
                        color=color)

        if len(x) > max_num_epochs:
            ax.set_xticks(x)
            if shrink_xtick_labels:
                xtick_labels = ['' if xi % 10 != 0 else xi for xi in x]
            else:
                xtick_labels = x
            ax.set_xticklabels(xtick_labels)
            max_num_epochs = len(x)

    plt.legend(frameon=False, fontsize=6)

    plt.tight_layout()
    return fig
