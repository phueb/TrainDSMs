from typing import Optional, List, Tuple
from pathlib import Path
import pandas as pd
from collections import defaultdict

from ludwig.results import gen_param_paths

from traindsms import __name__, config
from traindsms.figs import make_bar_plot
from traindsms.summary import save_summary_to_txt
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

WEAK_EVALUATION = False

# collect accuracies
gn2exp2c_accuracies = defaultdict(list)
project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         isolated=True if RUNS_PATH is not None else False,
                                         runs_path=RUNS_PATH,
                                         #ludwig_data_path=LUDWIG_DATA_PATH,
                                         label_n=LABEL_N,
                                         require_all_found=False,
                                         ):

    for p in param_path.rglob('df_sr.csv'):
        # read data
        df = pd.read_csv(p, index_col=0, squeeze=True)

        # exp 2c
        df_exp2c = df[(df['verb-type'] == 3) &
                      (df['theme-type'] == 'experimental') &
                      (df['phrase-type'] == 'unobserved')]

        hits = 0
        for verb_phrase, row in df_exp2c.iterrows():
            verb_phrase = verb_phrase.split()

        # strong evaluation requires that instruments be ranked correctly (1st and 2nd rank)

            if verb_phrase[0] == 'preserve':
                row_drop = row.drop(['vinegar','dehydrator'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['vinegar'] and other_max < row['dehydrator'])

            elif verb_phrase[0] == 'repair':
                row_drop = row.drop(['wrench', 'glue'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['wrench'] and other_max < row['glue'])

            elif verb_phrase[0] == 'pour':
                row_drop = row.drop(['pitcher', 'canister'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['pitcher'] and other_max < row['canister'])

            elif verb_phrase[0] == 'decorate':
                row_drop = row.drop(['icing', 'paint'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['icing'] and other_max < row['paint'])

            elif verb_phrase[0] == 'carve':
                row_drop = row.drop(['knife', 'chisel'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['knife'] and other_max < row['chisel'])

            elif verb_phrase[0] == 'heat':
                row_drop = row.drop(['oven', 'furnace'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['oven'] and other_max < row['furnace'])

            elif verb_phrase[0] == 'cut':
                row_drop = row.drop(['saw', 'scissors'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['saw'] and other_max < row['scissors'])

            elif verb_phrase[0] == 'clean':
                row_drop = row.drop(['towel', 'vacuum'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['towel'] and other_max < row['vacuum'])

            else:
                raise RuntimeError(f'Did not recognize verb-phrase "{verb_phrase}".')

        gn2exp2c_accuracies[label].append(hits / len(df_exp2c))

if not gn2exp2c_accuracies:
    raise SystemExit('Did not find results')

gn2exp2c_accuracies = {k: v for k, v in sorted(gn2exp2c_accuracies.items(), key=lambda i: sum(i[1]) / len(i[1]))}


for k, v in gn2exp2c_accuracies.items():
    print(k)
    print(v)
    print('-' * 32)

fig = make_bar_plot(gn2exp2c_accuracies,
                    ylabel='Exp2c Accuracy',)
fig.show()
