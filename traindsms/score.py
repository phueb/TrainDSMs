import pandas as pd


def score_vp_exp1(row: pd.Series,
                  verb: str,
                  theme: str,
                  ):
    if verb == 'grow':
        row_no_target = row.drop(['fertilizer'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['fertilizer'])

    elif verb == 'spray':
        row_no_target = row.drop(['insecticide'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['insecticide'])

    elif verb == 'fill':
        row_no_target = row.drop(['food'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['food'])

    elif verb == 'organize':
        row_no_target = row.drop(['organizer'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['organizer'])

    elif verb == 'freeze':
        row_no_target = row.drop(['freezer'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['freezer'])

    elif verb == 'consume':
        row_no_target = row.drop(['utensil'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['utensil'])

    elif verb == 'grill':
        row_no_target = row.drop(['bbq'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['bbq'])

    elif verb == 'catch':
        row_no_target = row.drop(['net'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['net'])

    elif verb == 'dry':
        row_no_target = row.drop(['dryer'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['dryer'])

    elif verb == 'dust':
        row_no_target = row.drop(['duster'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['duster'])

    elif verb == 'lubricate':
        row_no_target = row.drop(['lubricant'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['lubricant'])

    elif verb == 'seal':
        row_no_target = row.drop(['lacquer'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['lacquer'])

    elif verb == 'transfer':
        row_no_target = row.drop(['pump'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['pump'])


    elif verb == 'polish':
        row_no_target = row.drop(['polisher'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['polisher'])

    elif verb == 'shoot':
        row_no_target = row.drop(['slingshot'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['slingshot'])

    elif verb == 'harden':
        row_no_target = row.drop(['hammer'])
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['hammer'])

    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')


def score_vp_exp2a(row: pd.Series,
                   verb: str,
                   theme: str,
                   ):

    if verb == 'preserve':
        if theme == 'potato' or theme == 'cucumber':
            target = 'vinegar'
        else:
            target = 'dehydrator'
        row_no_target = row.drop(target)
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    elif verb == 'repair':
        if theme == 'fridge' or theme == 'microwave':
            target = 'wrench'
        else:
            target = 'glue'
        row_no_target = row.drop(target)
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    elif verb == 'pour':
        if theme == 'orange-juice' or theme == 'apple-juice':
            target = 'pitcher'
        else:
            target = 'canister'
        row_no_target = row.drop(target)
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    elif verb == 'decorate':
        if theme == 'pudding' or theme == 'pie':
            target = 'icing'
        else:
            target = 'paint'
        row_no_target = row.drop(target)
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    elif verb == 'carve':
        if theme == 'chicken' or theme == 'duck':
            target = 'knife'
        else:
            target = 'chisel'
        row_no_target = row.drop(target)
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    elif verb == 'heat':
        if theme == 'salmon' or theme == 'trout':
            target = 'oven'
        else:
            target = 'furnace'
        row_no_target = row.drop(target)
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    elif verb == 'cut':
        if theme == 'shirt' or theme == 'pants':
            target = 'scissors'
        else:
            target = 'saw'
        row_no_target = row.drop(target)
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    elif verb == 'clean':
        if theme == 'goggles' or theme == 'glove':
            target = 'towel'
        else:
            target = 'vacuum'
        row_no_target = row.drop(target)
        other_max = pd.to_numeric(row_no_target[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')


def score_vp_exp2b(row: pd.Series,
                   verb: str,
                   theme: str, ):
    """
    example (location-type=1):
    'preserve pepper' -> 'vinegar' (1) > 'dehydrator' (2) > others
    (1): 'cucumber'  is the SIBLING of 'pepper', and 'cucumber'  is preserved with 'vinegar'
    (2): 'raspberry' is the COUSIN  of 'pepper', and 'raspberry' is preserved with 'dehydrator'

    example (location-type=2):
    'pour tomato-juice' -> 'pitcher' (1) > 'canister' (2a) or 'freezer (2b) > others
    (1) : 'apple-juice' is the SIBLING of 'tomato-juice', and 'apple-juice' is poured with 'pitcher'
    (2a): 'coolant'     co-occurs with 'pour'             and 'coolant'     is poured with 'canister'
    (2b): 'freezer'     is the COUSIN  of 'tomato-juice', and 'freezer'     is frozen with 'freezer'

    when include_location=False, CTN slightly prefers (2a) over (2b)
    when include_location=True,  CTN slightly prefers (2b) over (2a) due to 'kitchen'
    """

    if verb == 'preserve':
        if theme == 'pepper':
            top1 = 'vinegar'
            top2 = 'dehydrator'
        else:
            top1 = 'dehydrator'
            top2 = 'vinegar'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    elif verb == 'repair':
        if theme == 'blender':
            top1 = 'wrench'
            top2 = 'glue'
        else:
            top1 = 'glue'
            top2 = 'wrench'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    elif verb == 'pour':
        if theme == 'tomato-juice':
            top1 = 'pitcher'
            top2 = 'canister'
        else:
            top1 = 'canister'
            top2 = 'pitcher'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    elif verb == 'decorate':
        if theme == 'cookie':
            top1 = 'icing'
            top2 = 'paint'
        else:
            top1 = 'paint'
            top2 = 'icing'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    elif verb == 'carve':
        if theme == 'turkey':
            top1 = 'knife'
            top2 = 'chisel'
        else:
            top1 = 'chisel'
            top2 = 'knife'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    elif verb == 'heat':
        if theme == 'tilapia':
            top1 = 'oven'
            top2 = 'furnace'
        else:
            top1 = 'furnace'
            top2 = 'oven'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    elif verb == 'cut':
        if theme == 'sock':
            top1 = 'scissors'
            top2 = 'saw'
        else:
            top1 = 'saw'
            top2 = 'scissors'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    elif verb == 'clean':
        if theme == 'faceshield':
            top1 = 'towel'
            top2 = 'vacuum'
        else:
            top1 = 'vacuum'
            top2 = 'towel'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')


def score_vp_exp2c(row: pd.Series,
                   verb: str,
                   theme: str, ):

    if verb == 'preserve':
        row_drop = row.drop(['vinegar', 'dehydrator'])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['vinegar'] and other_max < row['dehydrator'])

    elif verb == 'repair':
        row_drop = row.drop(['wrench', 'glue'])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['wrench'] and other_max < row['glue'])

    elif verb == 'pour':
        row_drop = row.drop(['pitcher', 'canister'])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['pitcher'] and other_max < row['canister'])

    elif verb == 'decorate':
        row_drop = row.drop(['icing', 'paint'])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['icing'] and other_max < row['paint'])

    elif verb == 'carve':
        row_drop = row.drop(['knife', 'chisel'])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['knife'] and other_max < row['chisel'])

    elif verb == 'heat':
        row_drop = row.drop(['oven', 'furnace'])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['oven'] and other_max < row['furnace'])

    elif verb == 'cut':
        row_drop = row.drop(['saw', 'scissors'])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['saw'] and other_max < row['scissors'])

    elif verb == 'clean':
        row_drop = row.drop(['towel', 'vacuum'])
        other_max = pd.to_numeric(row_drop[4:]).nlargest(n=1).to_list()[0]
        return int(other_max < row['towel'] and other_max < row['vacuum'])

    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')
