"""
save csv files with relative frequency of most confusable instruments in experiment 2a and 2b
"""
import dataclasses
from typing import Optional
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict, List

from ludwig.results import gen_param_paths

from traindsms import __name__
from traindsms import config
from traindsms.params import Params
from traindsms.params import param2default, param2requests

LUDWIG_DATA_PATH: Optional[Path] = None
RUNS_PATH = config.Dirs.runs  # config.Dirs.runs if loading runs locally or None if loading data from ludwig

LABEL_N: bool = True  # add information about number of replications to legend


records = []


@dataclasses.dataclass
class Counts:
    theme_type: str

    instrument_correct_count = 0
    instrument_type_3_sibling_category_count = 0
    instrument_type_2_same_category_count = 0
    instrument_type_2_sibling_category_count = 0
    instrument_other_count = 0

    num_total_evaluations = 0

    @property
    def instrument_correct_prop(self):
        return self.instrument_correct_count / self.num_total_evaluations

    @property
    def instrument_type_3_sibling_category_prop(self):
        return self.instrument_type_3_sibling_category_count / self.num_total_evaluations

    @property
    def instrument_type_2_same_category_prop(self):
        return self.instrument_type_2_same_category_count / self.num_total_evaluations

    @property
    def instrument_type_2_sibling_category_prop(self):
        return self.instrument_type_2_sibling_category_count / self.num_total_evaluations

    @property
    def instrument_other_prop(self):
        return self.instrument_other_count / self.num_total_evaluations


# map each verb phrase to the top 4 most confusable instruments.
# verb_phrase -> [correct, type 3 sibling category, type 2 same category, type 2 sibling category]

vp2instruments_exp2a = {
    # vegetable/fruit
    'preserve potato': ['vinegar', 'dehydrator', 'fertilizer', 'insecticide'],
    'preserve cucumber': ['vinegar', 'dehydrator', 'fertilizer', 'insecticide'],
    'preserve strawberry': ['dehydrator', 'vinegar', 'insecticide', 'fertilizer'],
    'preserve raspberry': ['dehydrator', 'vinegar', 'insecticide', 'fertilizer'],

    # appliance/kitchenware
    'repair fridge': ['wrench', 'glue', 'food', 'organizer'],
    'repair microwave': ['wrench', 'glue', 'food', 'organizer'],
    'repair plate': ['glue', 'wrench', 'organizer', 'food'],
    'repair cup': ['glue', 'wrench', 'organizer', 'food'],

    # juice/desert
    'cut shirt': ['scissors', 'saw', 'dryer', 'lacquer'],
    'cut pants': ['scissors', 'saw', 'dryer', 'lacquer'],
    'cut pine': ['saw', 'scissors', 'lacquer', 'dryer'],
    'cut mahogany': ['saw', 'scissors', 'lacquer', 'dryer'],

    # juice/desert
    'clean goggles': ['towel', 'vacuum', 'duster', 'lubricant'],
    'clean glove': ['towel', 'vacuum', 'duster', 'lubricant'],
    'clean tablesaw': ['vacuum', 'towel', 'lubricant', 'duster'],
    'clean beltsander': ['vacuum', 'towel', 'lubricant', 'duster'],
}


vp2instruments_exp2b = {
    # vegetable/fruit
    'preserve pepper': ['vinegar', 'dehydrator', 'fertilizer', 'insecticide'],
    'preserve orange': ['dehydrator', 'vinegar', 'insecticide', 'fertilizer'],

    # appliance/kitchenware
    'repair blender': ['wrench', 'glue', 'food', 'organizer'],
    'repair bowl': ['glue', 'wrench', 'organizer', 'food'],

    # juice/desert
    'cut sock': ['scissors', 'saw', 'dryer', 'lacquer'],
    'cut ash': ['saw', 'scissors', 'lacquer', 'dryer'],

    # juice/desert
    'clean faceshield': ['towel', 'vacuum', 'duster', 'lubricant'],
    'clean workstation': ['vacuum', 'towel', 'lubricant', 'duster'],
}



def categorize_predictions(counts: Counts,
                           vp2instruments: Dict[str, List[str]]) -> Counts:

    df_exp = df[(df['verb-type'] == 3) &
                (df['theme-type'] == counts.theme_type) &
                (df['phrase-type'] == 'observed') &
                (df['location-type'] == 1)]

    if df_exp.empty:
        raise RuntimeError('Did not find matching verb-theme combinations.')
    else:
        print(f'Extracted {len(df_exp):>3} rows of predictions for experiment 2 with them type {counts.theme_type}.')

    for verb_phrase, row in df_exp.iterrows():

        predictions = row[4:]  # predictions start after column 4

        instrument_predicted = predictions.astype(float).idxmax()

        try:
            instruments_top4 = vp2instruments[verb_phrase]
        except KeyError:
            # print(f'Could not find verb phrase "{verb_phrase}" in vp2instruments.')
            continue
        # else:
        #     print(f'Found verb phrase "{verb_phrase}" in vp2instruments.')


        try:
            idx = instruments_top4.index(instrument_predicted)
        except ValueError as ex:
            counts.instrument_other_count += 1
        else:
            if idx == 0:
                counts.instrument_correct_count += 1
            elif idx == 1:
                counts.instrument_type_3_sibling_category_count += 1
            elif idx == 2:
                counts.instrument_type_2_same_category_count += 1
            elif idx == 3:
                counts.instrument_type_2_sibling_category_count += 1
            else:
                raise RuntimeError('Encountered an unexpected index.')

        counts.num_total_evaluations += 1

    assert counts.num_total_evaluations > 0

    return counts


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

    # initialize counts
    num_total_evaluations = 0
    counts_exp_2a = Counts(theme_type='control')
    counts_exp_2b = Counts(theme_type='experimental')

    # for each replication/run
    for path_to_scores in param_path.rglob('df_sr.csv'):

        # read data
        df = pd.read_csv(path_to_scores, index_col=0, squeeze=True)

        # categorize predictions
        counts_exp_2a = categorize_predictions(counts=counts_exp_2a, vp2instruments=vp2instruments_exp2a)
        counts_exp_2b = categorize_predictions(counts=counts_exp_2b, vp2instruments=vp2instruments_exp2b)

    # add counts to records
    for counts in [counts_exp_2a, counts_exp_2b]:
        record = {
            'label': label,
            'dsm': params.dsm,
            'num_total_evaluations': counts.num_total_evaluations,
            'exp': {'control': '2a', 'experimental': '2b'}[counts.theme_type],

            'instrument_correct_prop': counts.instrument_correct_prop,
            'instrument_type_3_sibling_category_prop': counts.instrument_type_3_sibling_category_prop,
            'instrument_type_2_same_category_prop': counts.instrument_type_2_same_category_prop,
            'instrument_type_2_sibling_category_prop': counts.instrument_type_2_sibling_category_prop,
            'instrument_other_prop': counts.instrument_other_prop,
        }
        records.append(record)

df_out = pd.DataFrame.from_records(records).round(2)

# split the params into separate columns
df_params = df_out.copy()
df_params['label'] = df_params['label'].str.split('\n')
df_params = df_params.explode('label')
df_params[['param', 'value']] = df_params['label'].str.split('=', expand=True)
df_params = df_params.pivot(columns='param', values='value')

# concatenate the params with the results
df_out = pd.concat([df_params, df_out], axis=1)
df_out = df_out.drop(columns=['label'])

# sort
df_out = df_out.sort_values(by=['exp', 'instrument_correct_prop'], ascending=[True, False])


# save to disk
dsm = df_out["dsm"].values[0, 0]
path_out = (config.Dirs.data_for_analysis / f'exp2_confused_instruments_{dsm}.csv')
df_out.to_csv(path_out, index=False)

print(f'Saved {path_out}')

