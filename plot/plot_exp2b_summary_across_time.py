from typing import Optional, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from ludwig.results import gen_param_paths

from traindsms import __name__, config
from traindsms.figs import make_line_plot
from traindsms.score import score_vp_exp2b
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

WEAK_EVALUATION = 1

# collect accuracies
label2exp2b_accuracy_mat = {}
for param_path, label in gen_param_paths(__name__,
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

            # get items relevant to exp 2b
            df_exp2b = df[(df['verb-type'] == 3) &
                          (df['theme-type'] == 'experimental') &
                          (df['phrase-type'] == 'observed')]
            hits = 0
            for verb_phrase, row in df_exp2b.iterrows():

                verb, theme = verb_phrase.split()

                if WEAK_EVALUATION:  # a hit only requires that one instrument is scored greater than another

                    if verb_phrase == 'preserve pepper':
                        hits += int(row['vinegar'] > row['dehydrator'])
                    elif verb_phrase == 'preserve orange':
                        hits += int(row['dehydrator'] > row['vinegar'])
                    elif verb_phrase == 'repair blender':
                        hits += int(row['wrench'] > row['glue'])
                    elif verb_phrase == 'repair bowl':
                        hits += int(row['glue'] > row['wrench'])
                    elif verb_phrase == 'pour tomato-juice':
                        hits += int(row['pitcher'] > row['canister'])
                    elif verb_phrase == 'decorate cookie':
                        hits += int(row['icing'] > row['paint'])
                    elif verb_phrase == 'carve turkey':
                        hits += int(row['knife'] > row['chisel'])
                    elif verb_phrase == 'heat tilapia':
                        hits += int(row['oven'] > row['furnace'])
                    elif verb_phrase == 'cut sock':
                        hits += int(row['scissors'] > row['saw'])
                    elif verb_phrase == 'cut ash':
                        hits += int(row['saw'] > row['scissors'])
                    elif verb_phrase == 'clean faceshield':
                        hits += int(row['towel'] > row['vacuum'])
                    elif verb_phrase == 'clean workstation':
                        hits += int(row['vacuum'] > row['towel'])
                    elif verb_phrase == 'pour brake-fluid':
                        hits += int(row['canister'] > row['pitcher'])
                    elif verb_phrase == 'decorate motorcycle':
                        hits += int(row['paint'] > row['icing'])
                    elif verb_phrase == 'carve marble':
                        hits += int(row['chisel'] > row['knife'])
                    elif verb_phrase == 'heat copper':
                        hits += int(row['furnace'] > row['oven'])
                    else:
                        raise RuntimeError(f'Did not recognize verb-phrase "{verb_phrase}".')

                else:  # strong evaluation requires that instruments be ranked correctly (1st and 2nd rank)

                    hits += score_vp_exp2b(row, verb, theme)

            # collect accuracy
            acc_i = hits / len(df_exp2b)
            if label not in label2exp2b_accuracy_mat:
                num_epochs = len(epoch2dfs)
                num_reps = len(dfs)
                label2exp2b_accuracy_mat[label] = np.zeros((num_reps, num_epochs))
            label2exp2b_accuracy_mat[label][rep_id, epoch - 1] = acc_i  # -1 because epoch starts at 1



if not label2exp2b_accuracy_mat:
    raise SystemExit('Did not find results')

# sort
label2exp2b_accuracy_mat = {k: v for k, v in sorted(label2exp2b_accuracy_mat.items(), key=lambda i: np.mean(i[1][:, -1]))}


for k, v in label2exp2b_accuracy_mat.items():
    print(k)
    print(v[:, -1])
    print('-' * 32)


fig = make_line_plot(label2exp2b_accuracy_mat,
                     ylabel='Exp2b Accuracy',
                     h_line=0.5 if WEAK_EVALUATION else None)
fig.show()
