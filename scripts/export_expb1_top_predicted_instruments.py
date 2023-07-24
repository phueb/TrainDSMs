"""
save csv files with relative frequency of most confusable instruments in experiment 2b1
"""
from typing import Optional
from pathlib import Path
import pandas as pd
import yaml

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

records = []

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

    instrument_correct_count = 0
    instrument_type_3_sibling_category_count = 0
    instrument_type_2_same_category_count = 0
    instrument_type_2_sibling_category_count = 0
    instrument_other_count = 0

    num_total_evaluations = 0

    for path_to_scores in param_path.rglob('df_sr.csv'):

        # read data
        df = pd.read_csv(path_to_scores, index_col=0, squeeze=True)

        df_exp = df[(df['verb-type'] == VERB_TYPE) &
                    (df['theme-type'] == THEME_TYPE) &
                    (df['phrase-type'] == PHRASE_TYPE) &
                    (df['location-type'] == 1)]

        if df_exp.empty:
            raise RuntimeError('Did not find matching verb-phrase combinations.')
        else:
            print(f'Extracted {len(df_exp):>3} rows of predictions for experiment 2b1')

        for verb_phrase, row in df_exp.iterrows():

            verb, theme = verb_phrase.split()

            predictions = row[4:]  # predictions start after column 4

            # map each verb phrase to the top 4 most confusable instruments.
            # verb_phrase -> [correct, type 3 sibling category, type 2 same category, type 2 sibling category]
            vp2instruments = {
                'preserve pepper': ['vinegar', 'dehydrator', 'fertilizer', 'insecticide'],
                'preserve orange': ['dehydrator', 'vinegar', 'insecticide', 'fertilizer'],
            }

            instrument_predicted = predictions.astype(float).idxmax()

            try:
                instruments_top4 = vp2instruments[verb_phrase]
            except KeyError as ex:
                print(ex)
                continue

            try:
                idx = instruments_top4.index(instrument_predicted)
            except ValueError as ex:
                instrument_other_count += 1
            else:
                if idx == 0:
                    instrument_correct_count += 1
                elif idx == 1:
                    instrument_type_3_sibling_category_count += 1
                elif idx == 2:
                    instrument_type_2_same_category_count += 1
                elif idx == 3:
                    instrument_type_2_sibling_category_count += 1
                else:
                    raise RuntimeError('Encountered an unexpected index.')
            
            num_total_evaluations += 1
                
    path_out = (config.Dirs.data_for_analysis / param_path.name)
    if not path_out.exists():
        path_out.mkdir()

    instrument_correct_prop = instrument_correct_count / num_total_evaluations
    instrument_type_3_sibling_category_prop = instrument_type_3_sibling_category_count / num_total_evaluations
    instrument_type_2_same_category_prop = instrument_type_2_same_category_count / num_total_evaluations
    instrument_type_2_sibling_category_prop = instrument_type_2_sibling_category_count / num_total_evaluations
    instrument_other_prop = instrument_other_count / num_total_evaluations

    # create the row for this condition
    record = {
        'label': label,
        'param_name': param_path.name,
        'instrument_correct_prop': instrument_correct_prop,
        'instrument_type_3_sibling_category_prop': instrument_type_3_sibling_category_prop,
        'instrument_type_2_same_category_prop': instrument_type_2_same_category_prop,
        'instrument_type_2_sibling_category_prop': instrument_type_2_sibling_category_prop,
        'instrument_other_prop': instrument_other_prop,
    }

    records.append(record)

df_out = pd.DataFrame.from_records(records)

# save to disk
path_out = (config.Dirs.data_for_analysis / 'exp2b1_top_predicted_instruments.csv')
df_out.to_csv(path_out, index=False)

