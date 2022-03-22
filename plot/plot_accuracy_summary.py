from typing import Optional, List, Tuple
from pathlib import Path
import pandas as pd
import yaml
from collections import defaultdict

from ludwig.results import gen_param_paths

from traindsms import __name__
from traindsms.params import Params
from traindsms.figs import make_bar_plot
from traindsms.score import score_vp_exp1
from traindsms.score import score_vp_exp2a
from traindsms.score import score_vp_exp2b1
from traindsms.score import score_vp_exp2b2
from traindsms.score import score_vp_exp2c1
from traindsms.score import score_vp_exp2c2
from traindsms.score import score_vp_exp3b1
from traindsms.score import score_vp_exp3b2
from traindsms.score import score_vp_exp3c1
from traindsms.score import score_vp_exp3c2
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

experiments = [
    '1a',
    '1b',
    '1c',
    '2a',
    '2b1',
    '2b2',
    '2c1',
    '2c2',
    '3a1',
    '3a2',
    '3b1',
    '3b2',
    '3c1',
    '3c2',
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

    # get params
    with (param_path / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)
    params = Params.from_param2val(param2val)

    for p in param_path.rglob('df_sr.csv'):

        # read data
        df = pd.read_csv(p, index_col=0, squeeze=True)

        for exp in experiments:

            # some experiments require specific params
            if exp.startswith('1') and params.corpus_params.include_location:
                continue
            if exp.startswith('2') and params.corpus_params.include_location:
                continue
            if exp.startswith('3') and not params.corpus_params.include_location:
                continue

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
            elif exp == '2b1':
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'experimental') &
                            (df['phrase-type'] == 'observed') &
                            (df['location-type'] == 1)]
            elif exp == '2b2':
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'experimental') &
                            (df['phrase-type'] == 'observed') &
                            (df['location-type'] == 2)]
            elif exp == '2c1':
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'experimental') &
                            (df['phrase-type'] == 'unrelated') &  # unrelated as opposed to unobserved
                            (df['location-type'] == 1)]
            elif exp == '2c2':
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'experimental') &
                            (df['phrase-type'] == 'unrelated') &  # unrelated as opposed to unobserved
                            (df['location-type'] == 2)]

            elif exp == '3a1':
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'control') &
                            (df['phrase-type'] == 'observed') &
                            (df['location-type'] == 1)]
            elif exp == '3a2':
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'control') &
                            (df['phrase-type'] == 'observed') &
                            (df['location-type'] == 2)]
            elif exp == '3b1':
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'experimental') &
                            (df['phrase-type'] == 'observed') &
                            (df['location-type'] == 1)]
            elif exp == '3b2':
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'experimental') &
                            (df['phrase-type'] == 'observed') &
                            (df['location-type'] == 2)]
            elif exp == '3c1':
                df_exp = df[(df['verb-type'] == 3) &
                            (df['theme-type'] == 'experimental') &
                            (df['phrase-type'] == 'unrelated') &  # unrelated as opposed to unobserved
                            (df['location-type'] == 1)]
            elif exp == '3c2':
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

                # exp2 uses a different evaluation as exp1 but the training corpus is the same
                elif exp == '2a':
                    hits += score_vp_exp2a(predictions, verb, theme)
                elif exp == '2b1':
                    hits += score_vp_exp2b1(predictions, verb, theme)
                elif exp == '2b2':
                    hits += score_vp_exp2b2(predictions, verb, theme)
                elif exp == '2c1':
                    hits += score_vp_exp2c1(predictions, verb, theme)
                elif exp == '2c2':
                    hits += score_vp_exp2c2(predictions, verb, theme)

                # exp3 uses different evaluation as exp2b and a different training corpus
                elif exp == '3a1':
                    hits += score_vp_exp2a(predictions, verb, theme)
                elif exp == '3a2':
                    hits += score_vp_exp2a(predictions, verb, theme)
                elif exp == '3b1':
                    hits += score_vp_exp3b1(predictions, verb, theme)
                elif exp == '3b2':
                    hits += score_vp_exp3b2(predictions, verb, theme)
                elif exp == '3c1':
                    hits += score_vp_exp3c1(predictions, verb, theme)
                elif exp == '3c2':
                    hits += score_vp_exp3c2(predictions, verb, theme)
                else:
                    raise AttributeError(exp)

            # collect accuracy
            accuracy = hits / len(df_exp)
            exp2label2accuracies[exp].setdefault(label, []).append(accuracy)


for exp in experiments:

    try:
        label2accuracies = exp2label2accuracies[exp]
    except KeyError:
        raise KeyError(f'Did not find accuracies for experiment {exp}')

    # sort
    label2accuracies = {k: v for k, v in sorted(label2accuracies.items(), key=lambda i: sum(i[1]) / len(i[1]))}

    print_summaries(label2accuracies, exp)

    fig = make_bar_plot(label2accuracies,
                        ylabel=f'Experiment {exp} Accuracy',
                        )
    fig.show()
