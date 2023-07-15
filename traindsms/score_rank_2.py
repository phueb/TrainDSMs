import pandas as pd


def score_vp_exp2b1(predictions: pd.Series,
                    verb: str,
                    theme: str,
                    ):
    """
    a hit is recorded if both instruments that are correct in rank 2,
    are scored higher than all other instruments.
    the instrument that is correct in rank 1 is ignored.

    """

    verb_theme_dict = {
        'preserve': {
            'pepper': ['vinegar', 'dehydrator', 'fertilizer'],
            'orange': ['dehydrator', 'vinegar', 'insecticide'],
        },
        'repair': {
            'blender': ['wrench', 'glue', 'food'],
            'bowl': ['glue', 'wrench', 'organizer'],
        },
        'cut': {
            'sock': ['scissors', 'saw', 'dryer'],
            'ash': ['saw', 'scissors', 'lacquer'],
        },
        'clean': {
            'faceshield': ['towel', 'vacuum', 'duster'],
            'workstation': ['vacuum', 'towel', 'lubricant'],
        }
    }

    try:
        top1, top2a, top2b = verb_theme_dict[verb][theme]
        row_drop = predictions.drop([top1, top2a, top2b])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2a] and
                   other_max < predictions[top2b])
    except KeyError:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')