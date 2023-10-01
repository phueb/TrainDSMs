"""
save csv files showing which types of mistakes a model makes.
i.e. which kinds of instruments does a model predict when it makes an incorrect prediction?
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
    idx2name: Dict[int, str]
    exp: str
    theme_type: str

    # how often a prediction falls into a given category (there are 5 categories)
    idx0_count = 0
    idx1_count = 0
    idx2_count = 0
    idx3_count = 0
    idx4_count = 0

    num_total_evaluations = 0

    @property
    def idx0_prop(self):
        return self.idx0_count / self.num_total_evaluations

    @property
    def idx1_prop(self):
        return self.idx1_count / self.num_total_evaluations

    @property
    def idx2_prop(self):
        return self.idx2_count / self.num_total_evaluations

    @property
    def idx3_prop(self):
        return self.idx3_count / self.num_total_evaluations

    @property
    def idx4_prop(self):
        return self.idx4_count / self.num_total_evaluations


# map each verb phrase to the top 4 most confusable instruments.
# verb_phrase -> [correct, type 3 sibling category, type 3 same category, type 2 sibling category]

vp2instruments_exp1a = {
    # vegetable/fruit
    'grow potato': ['fertilizer', 'dehydrator', 'vinegar', 'insecticide'],
    'grow cucumber': ['fertilizer', 'dehydrator', 'vinegar', 'insecticide'],
    'spray strawberry': ['insecticide', 'vinegar', 'dehydrator', 'fertilizer'],
    'spray raspberry': ['insecticide', 'vinegar', 'dehydrator', 'fertilizer'],

    # appliance/kitchenware
    'fill fridge': ['food', 'glue', 'wrench', 'organizer'],
    'fill microwave': ['food', 'glue', 'wrench', 'organizer'],
    'organize plate': ['organizer', 'wrench', 'glue', 'food'],
    'organize cup': ['organizer', 'wrench', 'glue', 'food'],

    # juice/desert
    'dry shirt': ['dryer', 'saw', 'scissors', 'lacquer'],
    'dry pants': ['dryer', 'saw', 'scissors', 'lacquer'],
    'seal pine': ['lacquer', 'scissors', 'saw', 'dryer'],
    'seal mahogany': ['lacquer', 'scissors', 'saw', 'dryer'],

    # juice/desert
    'dust goggles': ['duster', 'vacuum', 'towel', 'lubricant'],
    'dust glove': ['duster', 'vacuum', 'towel', 'lubricant'],
    'lubricate tablesaw': ['lubricant', 'towel', 'vacuum', 'duster'],
    'lubricate beltsander': ['lubricant', 'towel', 'vacuum', 'duster'],
}

vp2instruments_exp1b = {
    # vegetable/fruit
    'grow pepper': ['fertilizer', 'dehydrator', 'vinegar', 'insecticide'],
    'spray orange': ['insecticide', 'vinegar', 'dehydrator', 'fertilizer'],

    # appliance/kitchenware
    'fill blender': ['food', 'glue', 'wrench', 'organizer'],
    'organize bowl': ['organizer', 'wrench', 'glue', 'food'],

    # juice/desert
    'dry sock': ['dryer', 'saw', 'scissors', 'lacquer'],
    'seal ash': ['lacquer', 'scissors', 'saw', 'dryer'],

    # juice/desert
    'dust faceshield': ['duster', 'vacuum', 'towel', 'lubricant'],
    'lubricate workstation': ['lubricant', 'towel', 'vacuum', 'duster'],
}

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



def categorize(df_exp: pd.DataFrame,
               counts: Counts,
               vp2instruments: Dict[str, List[str]]) -> Counts:

    if df_exp.empty:
        raise RuntimeError('Did not find matching verb-theme combinations.')
    else:
        print(f'Extracted {len(df_exp):>3} rows of predictions for experiment 2 with them type {counts.theme_type}.')

    for verb_phrase, row in df_exp.iterrows():

        predictions = row[4:]  # predictions start after column 4
        instrument_predicted = predictions.astype(float).idxmax()

        try:
            instruments_top4 = vp2instruments[str(verb_phrase)]
        except KeyError:
            # print(f'Could not find verb phrase "{verb_phrase}" in vp2instruments.')
            continue
        # else:
        #     print(f'Found verb phrase "{verb_phrase}" in vp2instruments.')


        try:
            idx = instruments_top4.index(instrument_predicted)
        except ValueError:
            counts.idx4_count += 1
        else:
            if idx == 0:
                counts.idx0_count += 1
            elif idx == 1:
                counts.idx1_count += 1
            elif idx == 2:
                counts.idx2_count += 1
            elif idx == 3:
                counts.idx3_count += 1
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


    idx2name_exp1 = {0: 'correct',
                     1: 'type_3_sibling_category',
                     2: 'type_3_same_category',
                     3: 'type_2_sibling_category',
                     4: 'other'}

    idx2name_exp2 = {0: 'correct',
                     1: 'type_3_sibling_category',
                     2: 'type_2_same_category',  # this is different from experiment 1
                     3: 'type_2_sibling_category',
                     4: 'other'}

    # initialize counts
    num_total_evaluations = 0
    counts_exp_1a = Counts(idx2name=idx2name_exp1, exp='1a', theme_type='control')
    counts_exp_1b = Counts(idx2name=idx2name_exp1, exp='1b', theme_type='experimental')
    counts_exp_2a = Counts(idx2name=idx2name_exp2, exp='2a', theme_type='control')
    counts_exp_2b = Counts(idx2name=idx2name_exp2, exp='2b', theme_type='experimental')

    # for each replication/run
    for path_to_scores in param_path.rglob('df_sr.csv'):

        # read data
        df = pd.read_csv(path_to_scores, index_col=0, squeeze=True)

        df_exp_1a = df[(df['verb-type'] == 2) &
                       (df['theme-type'] == counts_exp_1a.theme_type) &
                       (df['phrase-type'] == 'observed') &
                       (df['location-type'] == 0)]

        df_exp_1b = df[(df['verb-type'] == 2) &
                       (df['theme-type'] == counts_exp_1b.theme_type) &
                       (df['phrase-type'] == 'observed') &
                       (df['location-type'] == 0)]

        df_exp_2a = df[(df['verb-type'] == 3) &
                       (df['theme-type'] == counts_exp_2a.theme_type) &
                       (df['phrase-type'] == 'observed') &
                       (df['location-type'] == 1)]

        df_exp_2b = df[(df['verb-type'] == 3) &
                       (df['theme-type'] == counts_exp_2b.theme_type) &
                       (df['phrase-type'] == 'observed') &
                       (df['location-type'] == 1)]


        # categorize predictions
        counts_exp_1a = categorize(df_exp=df_exp_1a, counts=counts_exp_1a, vp2instruments=vp2instruments_exp1a)
        counts_exp_1b = categorize(df_exp=df_exp_1b, counts=counts_exp_1b, vp2instruments=vp2instruments_exp1b)
        counts_exp_2a = categorize(df_exp=df_exp_2a, counts=counts_exp_2a, vp2instruments=vp2instruments_exp2a)
        counts_exp_2b = categorize(df_exp=df_exp_2b, counts=counts_exp_2b, vp2instruments=vp2instruments_exp2b)

    # add counts to records
    for counts in [counts_exp_1a, counts_exp_1b, counts_exp_2a, counts_exp_2b]:
        record = {
            'label': label,
            'dsm': params.dsm,
            'num_total_evaluations': counts.num_total_evaluations,
            'exp': counts.exp,

            # proportion of predictions that falls into each category
            counts.idx2name[0]: counts.idx0_prop,
            counts.idx2name[1]: counts.idx1_prop,
            counts.idx2name[2]: counts.idx2_prop,
            counts.idx2name[3]: counts.idx3_prop,
            counts.idx2name[4]: counts.idx4_prop,
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
df_out['exp'] = df_out['exp'].astype(str)
df_out = df_out.sort_values(by=['exp', 'correct'], ascending=[True, False])


# save to disk
dsm = df_out["dsm"].values[0, 0]
path_out = (config.Dirs.data_for_analysis / f'confused_instruments_{dsm}.csv')
df_out.to_csv(path_out, index=False)

print(f'Saved {path_out}')

