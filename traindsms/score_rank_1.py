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
    """
    a hit is recorded if the correct instrument is scored highest.

    experiment 2a is different from 2b in that it uses only control, and not experimental, themes.
    """

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


def score_vp_exp2b1(predictions: pd.Series, verb: str, theme: str) -> int:
    """
    a hit is recorded if the correct instrument is scored highest, i.e. only rank 1 accuracy is considered.

    note: this scorer is relevant to location type = 1 only.
    """
    verb2theme2instrument = {
        'preserve': {
            'pepper': 'vinegar',
            'orange': 'dehydrator',
        },
        'repair': {
            'blender': 'wrench',
            'bowl': 'glue',
        },
        'cut': {
            'sock': 'scissors',
            'ash': 'saw',
        },
        'clean': {
            'faceshield': 'towel',
            'workstation': 'vacuum',
        }
    }
    try:
        theme2instrument = verb2theme2instrument[verb]
        instrument = theme2instrument[theme]
        row_drop_top1 = predictions.drop(instrument)
        other_max = row_drop_top1.max()
        return int(other_max < predictions[instrument])
    except KeyError:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')
