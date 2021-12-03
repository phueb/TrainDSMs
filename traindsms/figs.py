import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import sem, t
from typing import List, Tuple, Dict, Optional

from traindsms import config


def make_summary_fig(gn2accuracies: Dict[str, List[float]],
                     figsize: Tuple[int, int] = (8, 4),
                     title: str = '',
                     xlabel: str = 'Model',
                     ylabel: str = 'Accuracy',
                     width: float = 0.2,
                     confidence: float = 0.95,
                     y_grid: bool = False,
                     ylims: Optional[List[float]] = None,
                     h_line: Optional[float] = None,
                     ):
    # fig
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

    num_groups = len(gn2accuracies)
    edges = [width * i for i in range(num_groups)]  # x coordinate for each bar-center
    x = np.arange(1)

    # x-axis
    ax.set_xlabel(xlabel, fontsize=config.Figs.ax_font_size)
    ax.set_xticks(edges)
    if len(gn2accuracies) > 6:
        x_tick_labels = [gn if n == 0 or n == len(gn2accuracies) -1 else ''
                         for n, gn in enumerate(gn2accuracies)]
    else:
        x_tick_labels = gn2accuracies
    ax.set_xticklabels(x_tick_labels, fontsize=6)

    # y axis
    ax.set_ylabel(ylabel, fontsize=config.Figs.ax_font_size)

    # colors
    palette = np.asarray(sns.color_palette('hls', num_groups))
    colors = iter(palette)

    if h_line is not None:
        ax.axhline(y=h_line, color='grey', linestyle=':', zorder=1)

    # plot
    for edge, color, group_name in zip(edges, colors, gn2accuracies):
        accuracies = gn2accuracies[group_name]
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
