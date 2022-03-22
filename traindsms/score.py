import pandas as pd


def score_vp_exp1(predictions: pd.Series,
                  verb: str,
                  theme: str,
                  ) -> int:
    if verb == 'grow':
        row_no_target = predictions.drop(['fertilizer'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['fertilizer'])

    elif verb == 'spray':
        row_no_target = predictions.drop(['insecticide'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['insecticide'])

    elif verb == 'fill':
        row_no_target = predictions.drop(['food'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['food'])

    elif verb == 'organize':
        row_no_target = predictions.drop(['organizer'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['organizer'])

    elif verb == 'freeze':
        row_no_target = predictions.drop(['freezer'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['freezer'])

    elif verb == 'consume':
        row_no_target = predictions.drop(['utensil'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['utensil'])

    elif verb == 'grill':
        row_no_target = predictions.drop(['bbq'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['bbq'])

    elif verb == 'catch':
        row_no_target = predictions.drop(['net'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['net'])

    elif verb == 'dry':
        row_no_target = predictions.drop(['dryer'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['dryer'])

    elif verb == 'dust':
        row_no_target = predictions.drop(['duster'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['duster'])

    elif verb == 'lubricate':
        row_no_target = predictions.drop(['lubricant'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['lubricant'])

    elif verb == 'seal':
        row_no_target = predictions.drop(['lacquer'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['lacquer'])

    elif verb == 'transfer':
        row_no_target = predictions.drop(['pump'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['pump'])


    elif verb == 'polish':
        row_no_target = predictions.drop(['polisher'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['polisher'])

    elif verb == 'shoot':
        row_no_target = predictions.drop(['slingshot'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['slingshot'])

    elif verb == 'harden':
        row_no_target = predictions.drop(['hammer'])
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions['hammer'])

    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')


#############################################
# experiment 2
############################################


def score_vp_exp2a(predictions: pd.Series,
                   verb: str,
                   theme: str,
                   ) -> int:

    if verb == 'preserve':
        if theme == 'potato' or theme == 'cucumber':
            target = 'vinegar'
        else:
            target = 'dehydrator'
        row_no_target = predictions.drop(target)
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[target])

    elif verb == 'repair':
        if theme == 'fridge' or theme == 'microwave':
            target = 'wrench'
        else:
            target = 'glue'
        row_no_target = predictions.drop(target)
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[target])

    elif verb == 'pour':
        if theme == 'orange-juice' or theme == 'apple-juice':
            target = 'pitcher'
        else:
            target = 'canister'
        row_no_target = predictions.drop(target)
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[target])

    elif verb == 'decorate':
        if theme == 'pudding' or theme == 'pie':
            target = 'icing'
        else:
            target = 'paint'
        row_no_target = predictions.drop(target)
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[target])

    elif verb == 'carve':
        if theme == 'chicken' or theme == 'duck':
            target = 'knife'
        else:
            target = 'chisel'
        row_no_target = predictions.drop(target)
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[target])

    elif verb == 'heat':
        if theme == 'salmon' or theme == 'trout':
            target = 'oven'
        else:
            target = 'furnace'
        row_no_target = predictions.drop(target)
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[target])

    elif verb == 'cut':
        if theme == 'shirt' or theme == 'pants':
            target = 'scissors'
        else:
            target = 'saw'
        row_no_target = predictions.drop(target)
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[target])

    elif verb == 'clean':
        if theme == 'goggles' or theme == 'glove':
            target = 'towel'
        else:
            target = 'vacuum'
        row_no_target = predictions.drop(target)
        other_max = pd.to_numeric(row_no_target).nlargest(n=1).to_list()[0]
        return int(other_max < predictions[target])

    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')


def score_vp_exp2b1(predictions: pd.Series,
                    verb: str,
                    theme: str,
                    ) -> int:
    """
    example (location-type=1):
    'preserve pepper' -> 'vinegar' (1) > 'dehydrator' (2) > others
    (1): 'cucumber'  is the SIBLING of 'pepper', and 'cucumber'  is preserved with 'vinegar'
    (2): 'raspberry' is the COUSIN  of 'pepper', and 'raspberry' is preserved with 'dehydrator'
    """

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
        return int(other_max < predictions[top2] < predictions[top1])

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
        return int(other_max < predictions[top2] < predictions[top1])

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
        return int(other_max < predictions[top2] < predictions[top1])

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
        return int(other_max < predictions[top2] < predictions[top1])

    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')


def score_vp_exp2b2(predictions: pd.Series,
                    verb: str,
                    theme: str,
                    ) -> int:
    """
    example (location-type=2):
    'pour tomato-juice' -> 'pitcher' (1) > 'canister' (2a) or 'freezer (2b) > others
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
        return int(other_max < predictions[top2a] < predictions[top1] and
                   other_max < predictions[top2b] < predictions[top1])

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
        return int(other_max < predictions[top2a] < predictions[top1] and
                   other_max < predictions[top2b] < predictions[top1])

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
        return int(other_max < predictions[top2a] < predictions[top1] and
                   other_max < predictions[top2b] < predictions[top1])

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
        return int(other_max < predictions['pitcher'] and other_max < predictions['canister'])

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
# experiment 3
############################################


def score_vp_exp3b1(predictions: pd.Series,
                    verb: str,
                    theme: str,
                    ) -> int:
    return score_vp_exp2b1(predictions, verb, theme)


def score_vp_exp3b2(predictions: pd.Series,
                    verb: str,
                    theme: str,
                    ) -> int:
    """
    example (location-type=2):
    'pour tomato-juice' -> 'pitcher' (1) > 'freezer (2b) > others
    (1) : 'apple-juice' is the SIBLING of 'tomato-juice', and 'apple-juice' is poured with 'pitcher'
    (2) : 'freezer'     is the COUSIN  of 'tomato-juice', and 'freezer'     is frozen with 'freezer'

    when include_location=True, the ranking is unambiguous (there is a preferred 2nd rank that is location-specific)
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


def score_vp_exp3c1(predictions: pd.Series,
                    verb: str,
                    theme: str,
                    ) -> int:

    return score_vp_exp_c_base_(predictions, verb, theme)


def score_vp_exp3c2(predictions: pd.Series,
                    verb: str,
                    theme: str,
                    ) -> int:

    return score_vp_exp_c_base_(predictions, verb, theme)