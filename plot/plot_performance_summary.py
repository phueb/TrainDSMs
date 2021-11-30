from typing import Optional, List, Tuple
from pathlib import Path
import pandas as pd
from collections import defaultdict

from ludwig.results import gen_param_paths

from traindsms import __name__, config
from traindsms.figs import make_summary_fig
from traindsms.summary import save_summary_to_txt
from traindsms.params import param2default, param2requests

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH = config.Dirs.runs  # config.Dirs.runs if loading runs locally or None if loading data from ludwig

LABEL_N: bool = True  # add information about number of replications to legend
PLOT_MAX_LINE: bool = False  # plot horizontal line at best performance for each param
PLOT_MAX_LINES: bool = False  # plot horizontal line at best overall performance
PALETTE_IDS: Optional[List[int]] = None  # re-assign colors to each line
V_LINES: Optional[List[int]] = []  # add vertical lines to highlight time slices
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
CONFIDENCE: float = 0.95
TITLE = ''

# collect summaries
gn2exp2b_accuracies = defaultdict(list)
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

        # exp 2b
        df_exp2b = df[(df['verb-type'] == 3) &
                      (df['theme-type'] == 'experimental') &
                      (df['phrase-type'] == 'observed')]
        hits = 0
        for verb_phrase, row in df_exp2b.iterrows():
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
        gn2exp2b_accuracies[label].append(hits / len(df_exp2b))

    # save_summary_to_txt(summary, pattern)

print(gn2exp2b_accuracies)

fig = make_summary_fig(gn2exp2b_accuracies)
fig.show()
