"""
save csv files with scores for most confusable instruments in experiment 2b1
"""
from typing import Optional
from pathlib import Path
import pandas as pd
import yaml
import datetime
from collections import defaultdict

from ludwig.results import gen_param_paths

from traindsms import __name__
from traindsms import config
from traindsms.params import Params
from traindsms.params import param2default, param2requests

LUDWIG_DATA_PATH: Optional[Path] = None
RUNS_PATH = config.Dirs.runs  # config.Dirs.runs if loading runs locally or None if loading data from ludwig

LABEL_N: bool = True  # add information about number of replications to legend

VERB_TYPE = 3
THEME_TYPE = 'experimental'
PHRASE_TYPE = 'observed'

rank2label2accuracies = defaultdict(dict)

fertilizer = []
vinegar = []

wrench = []
food = []

scissors = []
dryer = []

towel = []
duster = []


for param_path, label in gen_param_paths(project_name=__name__,
                                         param2requests=param2requests,
                                         param2default=param2default,
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

    for path_to_scores in param_path.rglob('df_sr.csv'):

        # read data
        df = pd.read_csv(path_to_scores, index_col=0, squeeze=True)

        # exp 2b1 requires specific params
        # if params.corpus_params.include_location:
        #     continue

        df_exp = df[(df['verb-type'] == VERB_TYPE) &
                    (df['theme-type'] == THEME_TYPE) &
                    (df['phrase-type'] == PHRASE_TYPE) &
                    (df['location-type'] == 1)]

        print(f'Extracted {len(df_exp):>3} rows of predictions for experiment 2b1')

        for verb_phrase, row in df_exp.iterrows():

            verb, theme = verb_phrase.split()

            predictions = row[4:]  # predictions start after column 4

            # collect scores for the two most confusable instruments
            if verb_phrase == 'preserve pepper':
                vinegar.append(predictions['vinegar'])
                fertilizer.append(predictions['fertilizer'])
            if verb_phrase == 'repair blender':
                wrench.append(predictions['wrench'])
                food.append(predictions['food'])
            if verb_phrase == 'cut sock':
                scissors.append(predictions['scissors'])
                dryer.append(predictions['dryer'])
            if verb_phrase == 'clean faceshield':
                towel.append(predictions['towel'])
                duster.append(predictions['duster'])

    path_out = (config.Dirs.data_for_analysis / param_path.name)
    if not path_out.exists():
        path_out.mkdir()

    # save scores for the two most confusable instrument pairs, for analysis
    pd.DataFrame(data={
        'vinegar': vinegar,
        'fertilizer': fertilizer,
    }).to_csv(path_out / 'exp2b1_vinegar_fertilizer.csv', index=False)

    pd.DataFrame(data={
        'wrench': wrench,
        'food': food,
    }).to_csv(path_out / 'exp2b1_wrench_food.csv', index=False)

    pd.DataFrame(data={
        'scissors': scissors,
        'dryer': dryer,
    }).to_csv(path_out / 'exp2b1_scissors_dryer.csv', index=False)

    pd.DataFrame(data={
        'towel': towel,
        'duster': duster,
    }).to_csv(path_out / 'exp2b1_towel_duster.csv', index=False)

