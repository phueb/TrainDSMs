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
EXPERIMENT = '1a'  # 1a, 1b, or 1c

# collect accuracies
gn2exp1a_accuracies = defaultdict(list)
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

        if EXPERIMENT == '1a':
            df_exp1 = df[(df['verb-type'] == 2) &
                          (df['theme-type'] == 'control') &
                          (df['phrase-type'] == 'observed')]
        elif EXPERIMENT == '1b':
            df_exp1 = df[(df['verb-type'] == 2) &
                         (df['theme-type'] == 'experimental') &
                         (df['phrase-type'] == 'unobserved')]
        elif EXPERIMENT == '1c':
            df_exp1 = df[(df['verb-type'] == 2) &
                         (df['theme-type'] == 'experimental') &
                         (df['phrase-type'] == 'unobserved')]
        else:
            raise AttributeError('Invalid arg to EXPERIMENT')

        hits = 0
        for verb_phrase, row in df_exp1.iterrows():
            verb_phrase = verb_phrase.split()

            if verb_phrase[0] == 'grow':
                row_drop = row.drop(['fertilizer'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['fertilizer'])

            elif verb_phrase[0] == 'spray':
                row_drop = row.drop(['insecticide'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['insecticide'])

            elif verb_phrase[0] == 'fill':
                row_drop = row.drop(['food'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['food'])

            elif verb_phrase[0] == 'organize':
                row_drop = row.drop(['organizer'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['organizer'])

            elif verb_phrase[0] == 'freeze':
                row_drop = row.drop(['freezer'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['freezer'])

            elif verb_phrase[0] == 'consume':
                row_drop = row.drop(['utensil'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['utensil'])

            elif verb_phrase[0] == 'grill':
                row_drop = row.drop(['bbq'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['bbq'])

            elif verb_phrase[0] == 'catch':
                row_drop = row.drop(['net'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['net'])

            elif verb_phrase[0] == 'dry':
                row_drop = row.drop(['dryer'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['dryer'])

            elif verb_phrase[0] == 'dust':
                row_drop = row.drop(['duster'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['duster'])

            elif verb_phrase[0] == 'lubricate':
                row_drop = row.drop(['lubricant'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['lubricant'])

            elif verb_phrase[0] == 'seal':
                row_drop = row.drop(['lacquer'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['lacquer'])

            elif verb_phrase[0] == 'transfer':
                row_drop = row.drop(['pump'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['pump'])


            elif verb_phrase[0] == 'polish':
                row_drop = row.drop(['polisher'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['polisher'])

            elif verb_phrase[0] == 'shoot':
                row_drop = row.drop(['slingshot'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['slingshot'])

            elif verb_phrase[0] == 'harden':
                row_drop = row.drop(['hammer'])
                other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
                hits += int(other_max < row['hammer'])

            else:
                raise RuntimeError(f'Did not recognize verb-phrase "{verb_phrase}".')

        gn2exp1a_accuracies['test'].append(hits / len(df_exp1))



gn2exp1a_accuracies = {k: v for k, v in sorted(gn2exp1a_accuracies.items(), key=lambda i: sum(i[1]) / len(i[1]))}


for k, v in gn2exp1a_accuracies.items():
    print(k)
    print(v)
    print('-' * 32)

fig = make_bar_plot(gn2exp1a_accuracies)
fig.show()