from typing import Optional, List, Tuple
from pathlib import Path
import pandas as pd
from collections import defaultdict

from ludwig.results import gen_param_paths

from traindsms import __name__
from traindsms.figs import make_bar_plot
from traindsms.score import score_vp_exp2a
from traindsms.score import score_vp_exp2b
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

EXPERIMENT = '2a'  # 2a, 2b

# collect accuracies
gn2exp2_accuracies = defaultdict(list)
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

        if EXPERIMENT == '2a':
            df_exp2 = df[(df['verb-type'] == 3) &
                         (df['theme-type'] == 'control') &
                         (df['phrase-type'] == 'observed')]
        elif EXPERIMENT == '2b':
            df_exp2 = df[(df['verb-type'] == 3) &
                         (df['theme-type'] == 'experimental') &
                         (df['phrase-type'] == 'observed')]
        else:
            raise AttributeError('Invalid arg to EXPERIMENT')

        hits = 0
        for verb_phrase, row in df_exp2.iterrows():

            verb, theme = verb_phrase.split()

            if EXPERIMENT == '2a':
                hits += score_vp_exp2a(row, verb, theme)
            elif EXPERIMENT == '2b':
                hits += score_vp_exp2b(row, verb, theme)
            else:
                raise AttributeError('Invalid arg to EXPERIMENT')

        gn2exp2_accuracies[label].append(hits / len(df_exp2))

if not gn2exp2_accuracies:
    raise SystemExit('Did not find results')

# sort
gn2exp2_accuracies = {k: v for k, v in sorted(gn2exp2_accuracies.items(), key=lambda i: sum(i[1]) / len(i[1]))}

print_summaries(gn2exp2_accuracies)

fig = make_bar_plot(gn2exp2_accuracies,
                    ylabel=f'Exp {EXPERIMENT} Accuracy',
                    )
fig.show()
