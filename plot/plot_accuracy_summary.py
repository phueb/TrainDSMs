from typing import Optional, List, Tuple
from pathlib import Path
import pandas as pd
from collections import defaultdict

from ludwig.results import gen_param_paths

from traindsms import __name__
from traindsms.figs import make_bar_plot
from traindsms.score import score_vp_exp1
from traindsms.score import score_vp_exp2a
from traindsms.score import score_vp_exp2b
from traindsms.score import score_vp_exp2c
from traindsms.summary import print_summaries
from traindsms.params import param2default, param2requests

LUDWIG_DATA_PATH: Optional[Path] = None
RUNS_PATH = None  # config.Dirs.runs if loading runs locally or None if loading data from ludwig

LABEL_N: bool = True  # add information about number of replications to legend
PLOT_MAX_LINE: bool = False  # plot horizontal line at best performance for each param
PLOT_MAX_LINES: bool = False  # plot horizontal line at best overall performance
PALETTE_IDS: Optional[List[int]] = None  # re-assign colors to each line
V_LINES: Optional[List[int]] = []  # add vertical lines to highlight time slices
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
CONFIDENCE: float = 0.95
TITLE = ''

experiments = ['1a', '1b', '1c',
               '2a', '2b', '2c']

# collect accuracies
label2exp1a_accuracies = defaultdict(list)
label2exp1b_accuracies = defaultdict(list)
label2exp1c_accuracies = defaultdict(list)
label2exp2a_accuracies = defaultdict(list)
label2exp2b_accuracies = defaultdict(list)
label2exp2c_accuracies = defaultdict(list)
project_name = __name__

for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         isolated=True if RUNS_PATH is not None else False,
                                         runs_path=RUNS_PATH,
                                         ludwig_data_path=LUDWIG_DATA_PATH,
                                         label_n=LABEL_N,
                                         require_all_found=False,
                                         ):

    for p in param_path.rglob('df_sr.csv'):
        # read data
        df = pd.read_csv(p, index_col=0, squeeze=True)

        for exp in experiments:

            if exp == '1a':
                df_exp = df[(df['verb-type'] == 2) &
                            (df['theme-type'] == 'control') &
                            (df['phrase-type'] == 'observed')]
            elif exp == '1b':
                df_exp = df[(df['verb-type'] == 2) &
                            (df['theme-type'] == 'experimental') &
                            (df['phrase-type'] == 'observed')]
            elif exp == '1c':
                df_exp = df[(df['verb-type'] == 2) &
                            (df['theme-type'] == 'experimental') &
                            (df['phrase-type'] == 'unobserved')]

            elif exp == '2a':
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'control') &
                            (df['phrase-type'] == 'observed')]
            elif exp == '2b':
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'experimental') &
                            (df['phrase-type'] == 'observed')]
            elif exp == '2c':
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'experimental') &
                            (df['phrase-type'] == 'unobserved')]
            else:
                raise AttributeError(exp)

            # score
            hits = 0
            for verb_phrase, row in df_exp.iterrows():

                verb, theme = verb_phrase.split()

                if exp == '1a':
                    hits += score_vp_exp1(row, verb, theme)
                elif exp == '1b':
                    hits += score_vp_exp1(row, verb, theme)
                elif exp == '1c':
                    hits += score_vp_exp1(row, verb, theme)
                elif exp == '2a':
                    hits += score_vp_exp2a(row, verb, theme)
                elif exp == '2b':
                    hits += score_vp_exp2b(row, verb, theme)
                elif exp == '2c':
                    hits += score_vp_exp2c(row, verb, theme)
                else:
                    raise AttributeError(exp)

            # collect accuracy
            accuracy = hits / len(df_exp)
            if exp == '1a':
                label2exp1a_accuracies[label].append(accuracy)
            elif exp == '1b':
                label2exp1b_accuracies[label].append(accuracy)
            elif exp == '1c':
                label2exp1c_accuracies[label].append(accuracy)
            elif exp == '2a':
                label2exp2a_accuracies[label].append(accuracy)
            elif exp == '2b':
                label2exp2b_accuracies[label].append(accuracy)
            elif exp == '2c':
                label2exp2c_accuracies[label].append(accuracy)

for exp, label2accuracies in zip(experiments,
                                 [
                                     label2exp1a_accuracies,
                                     label2exp1b_accuracies,
                                     label2exp1c_accuracies,
                                     label2exp2a_accuracies,
                                     label2exp2b_accuracies,
                                     label2exp2c_accuracies,
                                 ]):

    if not label2accuracies:
        raise SystemExit('Did not find results')

    # sort
    label2accuracies = {k: v for k, v in sorted(label2accuracies.items(), key=lambda i: sum(i[1]) / len(i[1]))}

    print_summaries(label2accuracies, exp)

    fig = make_bar_plot(label2accuracies,
                        ylabel=f'Experiment {exp} Accuracy',
                        )
    fig.show()
