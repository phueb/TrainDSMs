import pandas as pd

from src import config

for shuffled in [False, True]:
    print('\n\nShuffled={}\n\n'.format(shuffled))
    # make 1 column per model
    cols = []
    time_of_inits = []
    for model_p in config.Dirs.runs.glob('*'):
        time_of_init = model_p.name
        print(time_of_init)
        col = []
        task_names = []
        # vertically stack all replications and tasks
        for scores_p in model_p.rglob('scores_*.csv'):
            task_name = scores_p.parent.name
            # get max across param configurations per task
            scores = pd.read_csv(scores_p, index_col=False)
            g = scores.groupby('shuffled')[['exp_score', 'nov_score']].max()
            col_chunk = g[g.index == shuffled]
            if not col_chunk.empty:
                col.append(col_chunk)
                task_names.append(task_name)
        # average over replications
        if col:
            col = pd.concat(col, axis=0).groupby(task_names).mean()
            print(col)
            cols.append(col)
            time_of_inits.append(time_of_init)
        print('==============================================================\n')
    # concatenate columns
    df = pd.concat(cols, axis=1, sort=True, keys=time_of_inits)  # passing keys results in hierarchical df
    print(df.to_string())
    df.index.name = 'task'
    df.to_csv('all_scores_{}.csv'.format('shuffled' if shuffled else ''))


