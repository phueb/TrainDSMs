import pandas as pd


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
        other_max = pd.to_numeric(row_no_target[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    elif verb == 'repair':
        if theme == 'fridge' or theme == 'microwave':
            target = 'wrench'
        else:
            target = 'glue'
        row_no_target = row.drop(target)
        other_max = pd.to_numeric(row_no_target[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    elif verb == 'pour':
        if theme == 'orange-juice' or theme == 'apple-juice':
            target = 'pitcher'
        else:
            target = 'canister'
        row_no_target = row.drop(target)
        other_max = pd.to_numeric(row_no_target[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    elif verb == 'decorate':
        if theme == 'pudding' or theme == 'pie':
            target = 'icing'
        else:
            target = 'paint'
        row_no_target = row.drop(target)
        other_max = pd.to_numeric(row_no_target[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    elif verb == 'carve':
        if theme == 'chicken' or theme == 'duck':
            target = 'knife'
        else:
            target = 'chisel'
        row_no_target = row.drop(target)
        other_max = pd.to_numeric(row_no_target[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    elif verb == 'heat':
        if theme == 'salmon' or theme == 'trout':
            target = 'oven'
        else:
            target = 'furnace'
        row_no_target = row.drop(target)
        other_max = pd.to_numeric(row_no_target[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    elif verb == 'cut':
        if theme == 'shirt' or theme == 'pants':
            target = 'scissors'
        else:
            target = 'saw'
        row_no_target = row.drop(target)
        other_max = pd.to_numeric(row_no_target[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    elif verb == 'clean':
        if theme == 'goggles' or theme == 'glove':
            target = 'towel'
        else:
            target = 'vacuum'
        row_no_target = row.drop(target)
        other_max = pd.to_numeric(row_no_target[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[target])

    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')


def score_vp_exp2b(row: pd.Series,
                   verb: str,
                   theme: str, ):
    if verb == 'preserve':
        if theme == 'pepper':
            top1 = 'vinegar'
            top2 = 'dehydrator'
        else:
            top1 = 'dehydrator'
            top2 = 'vinegar'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    elif verb == 'repair':
        if theme == 'blender':
            top1 = 'wrench'
            top2 = 'glue'
        else:
            top1 = 'glue'
            top2 = 'wrench'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    elif verb == 'pour':
        if theme == 'tomato-juice':
            top1 = 'pitcher'
            top2 = 'canister'
        else:
            top1 = 'canister'
            top2 = 'pitcher'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    elif verb == 'decorate':
        if theme == 'cookie':
            top1 = 'icing'
            top2 = 'paint'
        else:
            top1 = 'paint'
            top2 = 'icing'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    elif verb == 'carve':
        if theme == 'turkey':
            top1 = 'knife'
            top2 = 'chisel'
        else:
            top1 = 'chisel'
            top2 = 'knife'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    elif verb == 'heat':
        if theme == 'tilapia':
            top1 = 'oven'
            top2 = 'furnace'
        else:
            top1 = 'furnace'
            top2 = 'oven'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    elif verb == 'cut':
        if theme == 'sock':
            top1 = 'scissors'
            top2 = 'saw'
        else:
            top1 = 'saw'
            top2 = 'scissors'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    elif verb == 'clean':
        if theme == 'faceshield':
            top1 = 'towel'
            top2 = 'vacuum'
        else:
            top1 = 'vacuum'
            top2 = 'towel'
        row_drop = row.drop([top1, top2])
        other_max = pd.to_numeric(row_drop[3:]).nlargest(n=1).to_list()[0]
        return int(other_max < row[top2] < row[top1])

    else:
        raise RuntimeError(f'Did not recognize verb-phrase "{verb} {theme}".')


