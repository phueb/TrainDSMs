from typing import Optional, List, Tuple
from pathlib import Path
import pandas as pd
import yaml
from collections import defaultdict

from ludwig.results import gen_param_paths

from traindsms import __name__
from traindsms import config
from traindsms.params import Params
from traindsms.figs import make_bar_plot
from traindsms.score_rank_1 import score_vp_exp1
from traindsms.score_rank_1 import score_vp_exp2a
from traindsms.summary import print_summaries
from traindsms.params import param2default
from traindsms.params import param2requests

RANK_1_AND_2 = False

if RANK_1_AND_2:
    from traindsms.score_rank_1_and_2 import score_vp_exp2b1  # todo careful, this scores rank 1 and rank 2
else:
    from traindsms.score_rank_1 import score_vp_exp2b1

LUDWIG_DATA_PATH: Optional[Path] = None
RUNS_PATH = config.Dirs.runs  # config.Dirs.runs if loading runs locally or None if loading data from ludwig

LABEL_N: bool = True  # add information about number of replications to legend

experiments = [
    # '1a',
    # '1b',
    # '1c',

    # '2a',
    '2b1',
    # '2b2',
    # '2c1',
    # '2c2',
]


exp2label2accuracies = defaultdict(dict)
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

    label += f'\n{param_path.name}'

    # get params
    with (param_path / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)
    params = Params.from_param2val(param2val)

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

            elif exp.endswith('a'):
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'control') &
                            (df['phrase-type'] == 'observed')]
            elif exp.endswith('b1'):
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'experimental') &
                            (df['phrase-type'] == 'observed') &
                            (df['location-type'] == 1)]
            elif exp.endswith('b2'):
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'experimental') &
                            (df['phrase-type'] == 'observed') &
                            (df['location-type'] == 2)]
            elif exp.endswith('c1'):
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'experimental') &
                            (df['phrase-type'] == 'unrelated') &  # unrelated as opposed to unobserved
                            (df['location-type'] == 1)]
            elif exp.endswith('c2'):
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'experimental') &
                            (df['phrase-type'] == 'unrelated') &  # unrelated as opposed to unobserved
                            (df['location-type'] == 2)]

            else:
                raise AttributeError(exp)

            print(f'Extracted {len(df_exp):>3} rows of predictions for experiment {exp:<4}')

            # score
            hits = 0
            for verb_phrase, row in df_exp.iterrows():

                predictions = row[4:]  # predictions start after column 4

                verb, theme = verb_phrase.split()

                if exp == '1a':
                    hits += score_vp_exp1(predictions, verb, theme)
                elif exp == '1b':
                    hits += score_vp_exp1(predictions, verb, theme)
                elif exp == '1c':
                    hits += score_vp_exp1(predictions, verb, theme)

                # exp2
                elif exp in {'2a'}:
                    hits += score_vp_exp2a(predictions, verb, theme)
                elif exp in {'2b1'}:
                    hits += score_vp_exp2b1(predictions, verb, theme)
                elif exp in {'2b2'}:
                    hits += score_vp_exp2b2(predictions, verb, theme)
                elif exp in {'2c1'}:
                    hits += score_vp_exp2c1(predictions, verb, theme)
                elif exp in {'2c2'}:
                    hits += score_vp_exp2c2(predictions, verb, theme)

            # collect accuracy
            accuracy = hits / len(df_exp)
            exp2label2accuracies[exp].setdefault(label, []).append(accuracy)


for exp in experiments:

    label2accuracies = exp2label2accuracies[exp]

    if not label2accuracies:
        print(f'WARNING: Did not find accuracies for experiment {exp}')  # perhaps not all conditions were run?
        continue

    # sort
    label2accuracies = {k: v for k, v in sorted(label2accuracies.items(), key=lambda i: sum(i[1]) / len(i[1]))}

    # make colors consistent
    label2color_id = {k: n for n, k in enumerate(sorted(label2accuracies))}

    print_summaries(label2accuracies, exp)

    fig = make_bar_plot(label2accuracies,
                        ylabel=f'Experiment {exp} Accuracy',
                        # h_line_1=exp2chance_accuracy[exp.replace('3b', '2b')],
                        # h_line_2=1/12,
                        # h_line_3=1/6,
                        label2color_id=label2color_id
                        )
    fig.show()
