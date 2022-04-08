"""
with what accuracy does a model guess first rank correctly in experiment 2b1?
"""
from typing import Optional, List, Tuple
from pathlib import Path
import pandas as pd
import yaml
from collections import defaultdict


from ludwig.results import gen_param_paths

from traindsms import __name__
from traindsms.params import Params
from traindsms.figs import make_bar_plot
from traindsms.params import param2default, param2requests

LUDWIG_DATA_PATH: Optional[Path] = None
RUNS_PATH = None  # config.Dirs.runs if loading runs locally or None if loading data from ludwig

LABEL_N: bool = True  # add information about number of replications to legend


rank2label2accuracies = defaultdict(dict)
project_name = __name__


def score_rank1(verb, theme):
    if verb == 'preserve':
        # sibling
        if theme == 'pepper':
            top1 = 'vinegar'
            top2 = 'dehydrator'
        # cousin
        elif theme == 'orange':
            top1 = 'dehydrator'
            top2 = 'vinegar'
        else:
            raise RuntimeError
        row_drop = predictions.drop([top1])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top1])

    elif verb == 'repair':
        # sibling
        if theme == 'blender':
            top1 = 'wrench'
            top2 = 'glue'
        # cousin
        elif theme == 'bowl':
            top1 = 'glue'
            top2 = 'wrench'
        else:
            raise RuntimeError
        row_drop = predictions.drop([top1])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top1])

    elif verb == 'cut':
        # sibling
        if theme == 'sock':
            top1 = 'scissors'
            top2 = 'saw'
        # cousin
        elif theme == 'ash':
            top1 = 'saw'
            top2 = 'scissors'
        else:
            raise RuntimeError
        row_drop = predictions.drop([top1])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top1])

    elif verb == 'clean':
        # sibling
        if theme == 'faceshield':
            top1 = 'towel'
            top2 = 'vacuum'
        # cousin
        elif theme == 'workstation':
            top1 = 'vacuum'
            top2 = 'towel'
        else:
            raise RuntimeError
        row_drop = predictions.drop([top1])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top1])

    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')



def score_rank2(verb, theme):
    if verb == 'preserve':
        # sibling
        if theme == 'pepper':
            top1 = 'vinegar'
            top2 = 'dehydrator'
        # cousin
        elif theme == 'orange':
            top1 = 'dehydrator'
            top2 = 'vinegar'
        else:
            raise RuntimeError
        row_drop = predictions.drop([top1, top2])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2])

    elif verb == 'repair':
        # sibling
        if theme == 'blender':
            top1 = 'wrench'
            top2 = 'glue'
        # cousin
        elif theme == 'bowl':
            top1 = 'glue'
            top2 = 'wrench'
        else:
            raise RuntimeError
        row_drop = predictions.drop([top1, top2])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2])

    elif verb == 'cut':
        # sibling
        if theme == 'sock':
            top1 = 'scissors'
            top2 = 'saw'
        # cousin
        elif theme == 'ash':
            top1 = 'saw'
            top2 = 'scissors'
        else:
            raise RuntimeError
        row_drop = predictions.drop([top1, top2])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2])

    elif verb == 'clean':
        # sibling
        if theme == 'faceshield':
            top1 = 'towel'
            top2 = 'vacuum'
        # cousin
        elif theme == 'workstation':
            top1 = 'vacuum'
            top2 = 'towel'
        else:
            raise RuntimeError
        row_drop = predictions.drop([top1, top2])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2])

    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')


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

        # exp 2b1 requires specific params
        # if params.corpus_params.include_location:
        #     continue

        # exp 2b1
        df_exp = df[(df['verb-type'] == 3) &
                        (df['theme-type'] == 'experimental') &
                        (df['phrase-type'] == 'observed') &
                        (df['location-type'] == 1)]

        print(f'Extracted {len(df_exp):>3} rows of predictions for experiment 2b1')

        # score
        hits1 = 0
        hits2 = 0
        for verb_phrase, row in df_exp.iterrows():

            verb, theme = verb_phrase.split()

            predictions = row[4:]  # predictions start after column 4

            hits1 += score_rank1(verb, theme)
            hits2 += score_rank2(verb, theme)

        # collect accuracy
        accuracy1 = hits1 / len(df_exp)
        accuracy2 = hits2 / len(df_exp)
        rank2label2accuracies[1].setdefault(label, []).append(accuracy1)
        rank2label2accuracies[2].setdefault(label, []).append(accuracy2)


# plot
for rank in [1, 2]:

    label2accuracies = rank2label2accuracies[rank]

    # sort
    label2accuracies = {k: v for k, v in sorted(label2accuracies.items(), key=lambda i: sum(i[1]) / len(i[1]))}

    # make colors consistent
    label2color_id = {k: n for n, k in enumerate(sorted(label2accuracies))}


    fig = make_bar_plot(label2accuracies,
                        ylabel=f'Experiment 2b1 Rank {rank} Accuracy',
                        h_line_1=None,
                        h_line_2=1/4,
                        h_line_3=1/3,
                        label2color_id=label2color_id
                        )
    fig.show()
