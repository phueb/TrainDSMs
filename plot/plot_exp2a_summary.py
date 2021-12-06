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
gn2exp2a_accuracies = defaultdict(list)
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

        # exp 2a
        df_exp2a = df[(df['verb-type'] == 3) &
                      (df['theme-type'] == 'control') &
                      (df['phrase-type'] == 'observed')]

        hits = 0
        for verb_phrase, row in df_exp2a.iterrows():
            verb_phrase = verb_phrase.split()

            if verb_phrase[0] == 'preserve':
                if verb_phrase[1] == 'potato' or verb_phrase[1] == 'cucumber':
                    to_drop = 'vinegar'
                else:
                    to_drop = 'dehydrator'
                row_drop = row.drop(to_drop)
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row[to_drop])

            elif verb_phrase[0] == 'repair':
                if verb_phrase[1] == 'fridge' or verb_phrase[1] == 'microwave':
                    to_drop = 'wrench'
                else:
                    to_drop = 'glue'
                row_drop = row.drop(to_drop)
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row[to_drop])

            elif verb_phrase[0] == 'pour':
                if verb_phrase[1] == 'orange-juice' or verb_phrase[1] == 'apple-juice':
                    to_drop = 'pitcher'
                else:
                    to_drop = 'canister'
                row_drop = row.drop(to_drop)
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row[to_drop])

            elif verb_phrase[0] == 'decorate':
                if verb_phrase[1] == 'pudding' or verb_phrase[1] == 'pie':
                    to_drop = 'icing'
                else:
                    to_drop = 'paint'
                row_drop = row.drop(to_drop)
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row[to_drop])

            elif verb_phrase[0] == 'carve':
                if verb_phrase[1] == 'chicken' or verb_phrase[1] == 'duck':
                    to_drop = 'knife'
                else:
                    to_drop = 'chisel'
                row_drop = row.drop(to_drop)
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row[to_drop])

            elif verb_phrase[0] == 'heat':
                if verb_phrase[1] == 'salmon' or verb_phrase[1] == 'trout':
                    to_drop = 'oven'
                else:
                    to_drop = 'furnace'
                row_drop = row.drop(to_drop)
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row[to_drop])

            elif verb_phrase[0] == 'cut':
                if verb_phrase[1] == 'shirt' or verb_phrase[1] == 'pants':
                    to_drop = 'scissors'
                else:
                    to_drop = 'saw'
                row_drop = row.drop(to_drop)
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row[to_drop])


            elif verb_phrase[0] == 'clean':
                if verb_phrase[1] == 'goggles' or verb_phrase[1] == 'glove':
                    to_drop = 'towel'
                else:
                    to_drop = 'vacuum'
                row_drop = row.drop(to_drop)
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row[to_drop])

            else:
                raise RuntimeError(f'Did not recognize verb-phrase "{verb_phrase}".')

        gn2exp2a_accuracies[label].append(hits / len(df_exp2a))



gn2exp2a_accuracies = {k: v for k, v in sorted(gn2exp2a_accuracies.items(), key=lambda i: sum(i[1]) / len(i[1]))}


for k, v in gn2exp2a_accuracies.items():
    print(k)
    print(v)
    print('-' * 32)

fig = make_bar_plot(gn2exp2a_accuracies)
fig.show()