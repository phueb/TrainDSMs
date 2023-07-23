import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import sem, t
from typing import List, Tuple, Dict, Optional

from traindsms import config

# plt.rcParams['font.family'] = 'DeJavu Serif'
# plt.rcParams['font.serif'] = ['Times New Roman']

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "Computer Modern Roman"


def make_bar_plot(label2accuracies: Dict[str, List[float]],
                  figsize: Tuple[int, int] = (12, 6),
                  title: str = '',
                  xlabel: str = 'Model',
                  ylabel: str = 'Accuracy',
                  width: float = 0.2,
                  confidence: float = 0.95,
                  y_grid: bool = True,
                  ylims: Optional[List[float]] = None,
                  h_line_1: Optional[float] = None,
                  h_line_2: Optional[float] = None,
                  h_line_3: Optional[float] = None,
                  x_label_threshold: int = 10,
                  label2color_id: Dict[str, int] = None,
                  ):
    """
    plot average accuracy by group (job) at end of training.
    """

    if label2color_id is None:
        label2color_id = {label: n for n, label in enumerate(label2accuracies)}

    fig, ax = plt.subplots(figsize=figsize, dpi=config.Figs.dpi)
    plt.title(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    if y_grid:
        ax.yaxis.grid(True)
    if ylims is not None:
        ax.set_ylim(ylims)
        plt.yticks(np.arange(ylims[0], ylims[1], 0.1))  # increment y-ticks by 0.1
    else:
        ax.set_ylim([0.0, 1.05])
        plt.yticks(np.arange(0.0, 1.05, 0.1))  # increment y-ticks by 0.1

    num_groups = len(label2accuracies)
    edges = [width * i for i in range(num_groups)]  # x coordinate for each bar-center
    x = np.arange(1)

    # x-axis
    ax.set_xlabel(xlabel, fontsize=config.Figs.ax_font_size)
    ax.set_xticks(edges)
    if len(label2accuracies) > x_label_threshold:
        x_tick_labels = [label if n == 0 or n == len(label2accuracies) - 1 else ''
                         for n, label in enumerate(label2accuracies)]
    else:
        x_tick_labels = label2accuracies
    ax.set_xticklabels(x_tick_labels, fontsize=config.Figs.tick_font_size)

    # y axis
    ax.set_ylabel(ylabel, fontsize=config.Figs.ax_font_size)

    # colors
    palette = np.asarray(sns.color_palette('hls', num_groups))

    if h_line_1 is not None:
        ax.axhline(y=h_line_1, color='black', linestyle=':', zorder=3, label=f'guessing 1/{int(1/h_line_1)} correct')
    if h_line_2 is not None:
        ax.axhline(y=h_line_2, color='black', linestyle=':', zorder=3, label=f'guessing 1/{int(1/h_line_2)} correct')
    if h_line_3 is not None:
        ax.axhline(y=h_line_3, color='black', linestyle=':', zorder=3, label=f'guessing 1/{int(1/h_line_3)} correct')

    # plot
    for edge, label in zip(edges, label2accuracies):
        accuracies = label2accuracies[label]
        y = np.mean(accuracies)

        # margin of error (across paradigms, not replications)
        n = len(accuracies)
        h = sem(accuracies, axis=0) * t.ppf((1 + confidence) / 2, n - 1)  # margin of error

        # plot all bars belonging to a single model group (same color)
        rects = ax.bar(x + edge,
                       y,
                       width,
                       yerr=h,
                       color=palette[label2color_id[label]],
                       zorder=3,
                       )

        # annotate bar with mean accuracy
        if y < 0.1:
            continue
        horizontal_offset = rects[0].get_width() / 4
        ax.text(x=(rects[0].get_x() + rects[0].get_width() / 2) - horizontal_offset,
                y=y - 0.06,
                s=f'{y:.2f} ({h:.2f})',
                ha='center',
                va='bottom',
                fontsize=12,
                )

    plt.tight_layout()
    return fig


def make_line_plot(label2accuracy_mat: Dict[str, np.array],  # [num groups, num epochs]
                   fig_size: Tuple[int, int] = (8, 4),
                   title: str = '',
                   x_label: str = 'Epoch',
                   y_label: str = 'Accuracy',
                   confidence: float = 0.95,
                   y_grid: bool = False,
                   y_lims: Optional[List[float]] = None,
                   h_line: Optional[float] = None,
                   shrink_xtick_labels: bool = False,
                   label2color_id: Dict[str, int] = None,
                   ):
    """
    plot average accuracy by group across training.
    """

    if label2color_id is None:
        label2color_id = {label: n for n, label in enumerate(label2accuracy_mat)}

    fig, ax = plt.subplots(figsize=fig_size, dpi=config.Figs.dpi)
    plt.title(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    if y_grid:
        ax.yaxis.grid(True)
    if y_lims is not None:
        ax.set_ylim(y_lims)
    else:
        ax.set_ylim([0.0, 1.05])

    # axes
    ax.set_xlabel(x_label, fontsize=config.Figs.ax_font_size)
    ax.set_ylabel(y_label, fontsize=config.Figs.ax_font_size)

    # colors
    num_groups = len(label2accuracy_mat)
    palette = np.asarray(sns.color_palette('hls', num_groups))

    if h_line is not None:
        ax.axhline(y=h_line, color='grey', linestyle=':', zorder=1)

    # plot
    max_num_epochs = 1
    for label in label2accuracy_mat:
        mat = label2accuracy_mat[label]
        y_mean = np.mean(mat, axis=0)
        x = np.arange(len(y_mean))

        # margin of error (across paradigms, not replications)
        n = len(mat)
        h = sem(mat, axis=0) * t.ppf((1 + confidence) / 2, n - 1)  # margin of error

        # plot
        ax.plot(x, y_mean, '-',
                color=palette[label2color_id[label]],
                label=label,
                )
        ax.fill_between(x,
                        y_mean + h,
                        y_mean - h,
                        alpha=0.2,
                        color=palette[label2color_id[label]],
                        )

        if len(x) > max_num_epochs:
            ax.set_xticks(x)
            if shrink_xtick_labels:
                xtick_labels = ['' if xi % 10 != 0 else xi for xi in x]
            else:
                xtick_labels = x
            ax.set_xticklabels(xtick_labels, fontsize=config.Figs.tick_font_size)
            max_num_epochs = len(x)

    plt.legend(bbox_to_anchor=(0.5, 1.0),
               borderaxespad=1.0,
               fontsize=6,
               frameon=False,
               loc='lower center',
               ncol=6,
               )

    plt.tight_layout()
    return fig
