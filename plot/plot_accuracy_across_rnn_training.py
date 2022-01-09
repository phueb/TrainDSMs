from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

from ludwig.results import gen_param_paths

from traindsms import __name__
from traindsms.figs import make_line_plot
from traindsms.score import score_vp_exp1
from traindsms.score import score_vp_exp2a
from traindsms.score import score_vp_exp2b
from traindsms.score import score_vp_exp2c
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
label2exp1a_accuracy_mat = {}
label2exp1b_accuracy_mat = {}
label2exp1c_accuracy_mat = {}
label2exp2a_accuracy_mat = {}
label2exp2b_accuracy_mat = {}
label2exp2c_accuracy_mat = {}
project_name = __name__


def update_accuracy_mat(label_: str,
                        label2mat: Dict[str, Any],
                        val: float,
                        i: int,
                        j: float,
                        ):
    # initialize matrix with zeros
    if label_ not in label2mat:
        label2mat[label_] = np.zeros((num_reps, num_epochs))

    label2mat[label_][i, j - 1] = val  # -1 because epoch starts at 1

    return label2mat


for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         isolated=True if RUNS_PATH is not None else False,
                                         runs_path=RUNS_PATH,
                                         ludwig_data_path=LUDWIG_DATA_PATH,
                                         label_n=LABEL_N,
                                         require_all_found=False,
                                         ):
    # read data
    epoch2dfs = defaultdict(list)
    num_found = 0
    for p in param_path.rglob('df_sr_*.csv'):
        df = pd.read_csv(p, index_col=0, squeeze=True)
        epoch = int(p.stem.split('_')[-1])
        epoch2dfs[epoch].append(df)
        num_found += 1
    if num_found == 0:
        raise RuntimeError('Did not find files matching "df_sr_*.csv"')

    for epoch, dfs in epoch2dfs.items():

        # for each replication/job
        for rep_id, df in enumerate(dfs):

            # for each experiment
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
                acc_i = hits / len(df_exp)
                num_epochs = len(epoch2dfs)
                num_reps = len(dfs)
                if exp == '1a':
                    update_accuracy_mat(label, label2exp1a_accuracy_mat, acc_i, rep_id, epoch)  # TODO need to return mat?
                elif exp == '1b':
                    update_accuracy_mat(label, label2exp1b_accuracy_mat, acc_i, rep_id, epoch)
                elif exp == '1c':
                    update_accuracy_mat(label, label2exp1c_accuracy_mat, acc_i, rep_id, epoch)
                elif exp == '2a':
                    update_accuracy_mat(label, label2exp2a_accuracy_mat, acc_i, rep_id, epoch)
                elif exp == '2b':
                    update_accuracy_mat(label, label2exp2b_accuracy_mat, acc_i, rep_id, epoch)
                elif exp == '2c':
                    update_accuracy_mat(label, label2exp2c_accuracy_mat, acc_i, rep_id, epoch)
                else:
                    raise AttributeError(exp)

for exp, label2accuracy_mat in zip(experiments,
                                   [
                                       label2exp1a_accuracy_mat,
                                       label2exp1b_accuracy_mat,
                                       label2exp1c_accuracy_mat,
                                       label2exp2a_accuracy_mat,
                                       label2exp2b_accuracy_mat,
                                       label2exp2c_accuracy_mat,
                                   ]):

    if not label2accuracy_mat:
        raise SystemExit('Did not find results')

    # sort
    label2accuracy_mat = {k: v for k, v in sorted(label2accuracy_mat.items(), key=lambda i: np.mean(i[1][:, -1]))}

    fig = make_line_plot(label2accuracy_mat,
                         ylabel=f'Experiment {exp} Accuracy',
                         )
    fig.show()
