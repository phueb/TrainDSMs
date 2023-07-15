import pandas as pd


# chance-level was computed using the random-control dsm (producing random relatedness scores)
exp2chance_accuracy = {
    '1a':  0.0294,
    '1b':  0.0238,
    '1c':  0.0200,

    '2a':  0.0312,
    '2b1': 0.0010,  # formally computed
    '2b2': 0.0010,
    '2c1': 0.0036,
    '2c2': 0.0027,

}


def score_vp_exp1(predictions: pd.Series,
                  verb: str,
                  theme: str,
                  ) -> int:
    """
    a hit is recorded if the correct instrument is scored highest.
    """

    verb2instrument = {
        'grow': 'fertilizer',
        'spray': 'insecticide',
        'fill': 'food',
        'organize': 'organizer',
        'freeze': 'freezer',
        'consume': 'utensil',
        'grill': 'bbq',
        'catch': 'net',
        'dry': 'dryer',
        'dust': 'duster',
        'lubricate': 'lubricant',
        'seal': 'lacquer',
        'transfer': 'pump',
        'polish': 'polisher',
        'shoot': 'slingshot',
        'harden': 'hammer'
    }

    if verb in verb2instrument:
        instrument = verb2instrument[verb]
        row_no_target = predictions.drop(instrument)
        other_max = row_no_target.max()
        return int(other_max < predictions[instrument])
    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')


def score_vp_exp2a(predictions: pd.Series, verb: str, theme: str) -> int:
    verb2theme2instrument = {
        'preserve': {
            'potato': 'vinegar',
            'cucumber': 'vinegar',
            'strawberry': 'dehydrator',
            'raspberry': 'dehydrator'
        },
        'repair': {
            'fridge': 'wrench',
            'microwave': 'wrench',
            'plate': 'glue',
            'cup': 'glue'
        },
        'pour': {
            'orange-juice': 'pitcher',
            'apple-juice': 'pitcher',
            'coolant': 'canister',
            'anti-freeze': 'canister'
        },
        'decorate': {
            'pudding': 'icing',
            'pie': 'icing',
            'car': 'paint',
            'truck': 'paint'
        },
        'carve': {
            'chicken': 'knife',
            'duck': 'knife',
            'granite': 'chisel',
            'limestone': 'chisel'
        },
        'heat': {
            'salmon': 'oven',
            'trout': 'oven',
            'iron': 'furnace',
            'steel': 'furnace'
        },
        'cut': {
            'shirt': 'scissors',
            'pants': 'scissors',
            'pine': 'saw',
            'mahogany': 'saw'
        },
        'clean': {
            'goggles': 'towel',
            'glove': 'towel',
            'tablesaw': 'vacuum',
            'beltsander': 'vacuum'
        }
    }
    try:
        theme2instrument = verb2theme2instrument[verb]
        target = theme2instrument[theme]
        row_no_target = predictions.drop(target)
        other_max = row_no_target.max()
        return int(other_max < predictions[target])
    except KeyError:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')


def score_vp_exp2b1(predictions: pd.Series,
                    verb: str,
                    theme: str,
                    ) -> int:
    """
    example (location-type=1):
    'preserve pepper': 'vinegar' (1) > 'dehydrator' (2a) or 'fertilizer' (2b) > others
    (1): 'cucumber'  is the SIBLING of 'pepper', and 'cucumber'  is preserved with 'vinegar'
    (2a): 'raspberry' is the COUSIN  of 'pepper', and 'raspberry' is preserved with 'dehydrator'
    (2b): 'cucumber' is the SIBLING  of 'pepper', and 'cucumber' is grown with 'fertilizer'
    """

    if verb == 'preserve':
        if theme == 'pepper':
            top1 = 'vinegar'
            top2a = 'dehydrator'
            top2b = 'fertilizer'
        elif theme == 'orange':
            top1 = 'dehydrator'
            top2a = 'vinegar'
            top2b = 'insecticide'
        else:
            raise SystemExit
        row_drop = predictions.drop([top1, top2a, top2b])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2a] < predictions[top1] and
                   other_max < predictions[top2b] < predictions[top1])

    elif verb == 'repair':
        if theme == 'blender':
            top1 = 'wrench'
            top2a = 'glue'
            top2b = 'food'
        elif theme == 'bowl':
            top1 = 'glue'
            top2a = 'wrench'
            top2b = 'organizer'
        else:
            raise SystemExit
        row_drop = predictions.drop([top1, top2a, top2b])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2a] < predictions[top1] and
                   other_max < predictions[top2b] < predictions[top1])

    elif verb == 'cut':
        if theme == 'sock':
            top1 = 'scissors'
            top2a = 'saw'
            top2b = 'dryer'
        elif theme == 'ash':
            top1 = 'saw'
            top2a = 'scissors'
            top2b = 'lacquer'
        else:
            raise SystemExit
        row_drop = predictions.drop([top1, top2a, top2b])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2a] < predictions[top1] and
                   other_max < predictions[top2b] < predictions[top1])

    elif verb == 'clean':
        if theme == 'faceshield':
            top1 = 'towel'
            top2a = 'vacuum'
            top2b = 'duster'
        elif theme == 'workstation':
            top1 = 'vacuum'
            top2a = 'towel'
            top2b = 'lubricant'
        else:
            raise SystemExit
        row_drop = predictions.drop([top1, top2a, top2b])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2a] < predictions[top1] and
                   other_max < predictions[top2b] < predictions[top1])

    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')


def score_vp_exp2b2(predictions: pd.Series,
                    verb: str,
                    theme: str,
                    ) -> int:
    """
    example (location-type=2):
    'pour tomato-juice': 'pitcher' (1) > 'canister' (2a) or 'freezer (2b) > others
    (1) : 'apple-juice' is the SIBLING of 'tomato-juice', and 'apple-juice' is poured with 'pitcher'
    (2a): 'coolant'     co-occurs with 'pour'             and 'coolant'     is poured with 'canister'
    (2b): 'freezer'     is the COUSIN  of 'tomato-juice', and 'freezer'     is frozen with 'freezer'

    when include_location=False, CTN slightly prefers (2a) over (2b)
    when include_location=True,  CTN slightly prefers (2b) over (2a) due to 'kitchen'
    """

    if verb == 'pour':
        if theme == 'tomato-juice':
            top1 = 'pitcher'
            top2a = 'canister'
            top2b = 'freezer'
        elif theme == 'brake-fluid':
            top1 = 'canister'
            top2a = 'pitcher'
            top2b = 'pump'
        else:
            raise SystemExit
        row_drop = predictions.drop([top1, top2a, top2b])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2a] < predictions[top1] and
                   other_max < predictions[top2b] < predictions[top1])

    elif verb == 'decorate':
        if theme == 'cookie':
            top1 = 'icing'
            top2a = 'paint'
            top2b = 'utensil'
        elif theme == 'motorcycle':
            top1 = 'paint'
            top2a = 'icing'
            top2b = 'polisher'
        else:
            raise SystemExit
        row_drop = predictions.drop([top1, top2a, top2b])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2b] < predictions[top2a] < predictions[top1] or
                   other_max < predictions[top2a] < predictions[top2b] < predictions[top1])

    elif verb == 'carve':
        if theme == 'turkey':
            top1 = 'knife'
            top2a = 'chisel'
            top2b = 'bbq'
        elif theme == 'marble':
            top1 = 'chisel'
            top2a = 'knife'
            top2b = 'slingshot'
        else:
            raise SystemExit
        row_drop = predictions.drop([top1, top2a, top2b])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2b] < predictions[top2a] < predictions[top1] or
                   other_max < predictions[top2a] < predictions[top2b] < predictions[top1])

    elif verb == 'heat':
        if theme == 'tilapia':
            top1 = 'oven'
            top2a = 'furnace'
            top2b = 'net'
        elif theme == 'copper':
            top1 = 'furnace'
            top2a = 'oven'
            top2b = 'hammer'
        else:
            raise SystemExit
        row_drop = predictions.drop([top1, top2a, top2b])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2b] < predictions[top2a] < predictions[top1] or
                   other_max < predictions[top2a] < predictions[top2b] < predictions[top1])

    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')


def score_vp_exp2c1(predictions: pd.Series,
                    verb: str,
                    theme: str,
                    ) -> int:

    return score_vp_exp_c_base_(predictions, verb, theme)


def score_vp_exp2c2(predictions: pd.Series,
                    verb: str,
                    theme: str,
                    ) -> int:

    return score_vp_exp_c_base_(predictions, verb, theme)


def score_vp_exp_c_base_(predictions: pd.Series,
                         verb: str,
                         theme: str,
                         ) -> int:
    """
    how well does a model infer the correct instrument when the only useful information is the verb?
    """

    if verb == 'preserve':
        row_drop = predictions.drop(['vinegar', 'dehydrator'])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['vinegar'] and other_max < predictions['dehydrator'])

    elif verb == 'repair':
        row_drop = predictions.drop(['wrench', 'glue'])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['wrench'] and other_max < predictions['glue'])

    elif verb == 'pour':
        row_drop = predictions.drop(['pitcher', 'canister'])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['pitcher'] and other_max < predictions['canister'])  # TODO in 5c2, there should be a clear ranking

    elif verb == 'decorate':
        row_drop = predictions.drop(['icing', 'paint'])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['icing'] and other_max < predictions['paint'])

    elif verb == 'carve':
        row_drop = predictions.drop(['knife', 'chisel'])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['knife'] and other_max < predictions['chisel'])

    elif verb == 'heat':
        row_drop = predictions.drop(['oven', 'furnace'])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['oven'] and other_max < predictions['furnace'])

    elif verb == 'cut':
        row_drop = predictions.drop(['saw', 'scissors'])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['saw'] and other_max < predictions['scissors'])

    elif verb == 'clean':
        row_drop = predictions.drop(['towel', 'vacuum'])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['towel'] and other_max < predictions['vacuum'])

    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')

#############################################
# experiment 5
############################################


def score_vp_exp5b1(predictions: pd.Series,
                    verb: str,
                    theme: str,
                    ) -> int:
    return score_vp_exp2b1(predictions, verb, theme)


def score_vp_exp5b2(predictions: pd.Series,
                    verb: str,
                    theme: str,
                    ) -> int:
    """
    example (location-type=2):
    'pour tomato-juice' -> 'pitcher' (1) > 'freezer (2b) > others
    (1) : 'apple-juice' is the SIBLING of 'tomato-juice', and 'apple-juice' is poured with 'pitcher'
    (2) : 'freezer'     is the COUSIN  of 'tomato-juice', and 'freezer'     is frozen with 'freezer'

    when include_location=True, the ranking is unambiguous (there is a preferred 2nd rank that is location-specific)

    WARNING:
        the accuracies in exp 2b2 and 5b2 are NOT comparable because a different target ranking is used

    """

    if verb == 'pour':
        if theme == 'tomato-juice':
            top1 = 'pitcher'
            top2 = 'freezer'
        elif theme == 'brake-fluid':
            top1 = 'canister'
            top2 = 'pump'
        else:
            raise SystemExit
        row_drop = predictions.drop([top1, top2])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2] < predictions[top1])

    elif verb == 'decorate':
        if theme == 'cookie':
            top1 = 'icing'
            top2 = 'utensil'
        elif theme == 'motorcycle':
            top1 = 'paint'
            top2 = 'polisher'
        else:
            raise SystemExit
        row_drop = predictions.drop([top1, top2])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2] < predictions[top1])

    elif verb == 'carve':
        if theme == 'turkey':
            top1 = 'knife'
            top2 = 'bbq'
        elif theme == 'marble':
            top1 = 'chisel'
            top2 = 'slingshot'
        else:
            raise SystemExit
        row_drop = predictions.drop([top1, top2])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2] < predictions[top1])

    elif verb == 'heat':
        if theme == 'tilapia':
            top1 = 'oven'
            top2 = 'net'
        elif theme == 'copper':
            top1 = 'furnace'
            top2 = 'hammer'
        else:
            raise SystemExit
        row_drop = predictions.drop([top1, top2])
        other_max = pd.to_numeric(row_drop).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[top2] < predictions[top1])

    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')

