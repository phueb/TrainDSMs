import argparse
from pathlib import Path

from src import config
from src.jobs import embedder_job, aggregation_job
from src.params import params_df


def run_on_cluster():
    """
    run multiple jobs on multiple LudwigCluster nodes.
    """
    for n, params_df_row in params_df.iterrows():
        config.Eval.num_reps = params_df_row['num_reps']
        embedder_name = params_df_row['embedder_name']
        print('Training and evaluating "{}" with num_reps={}.'.format(embedder_name, config.Eval.num_reps))
        # overwrite directories where data is stored and saved
        config.Dirs.runs = Path(params_df_row['runs_dir'])
        config.Dirs.corpora = Path(params_df_row['runs_dir'].replace('runs', 'corpora'))
        config.Dirs.tasks = Path(params_df_row['runs_dir'].replace('runs', 'tasks'))
        embedder_job(embedder_name, params_df_row=params_df_row)
    #
    try:
        aggregation_job('matching')
        aggregation_job('identification')
    except RuntimeError:  # did not find any scores for identification
        pass
    print('Done')


def run_on_host(params_df):
    """
    run jobs on the local host for testing/development
    """
    for n, params_df_row in params_df.iterrows():
        embedder_name = params_df_row['embedder_name']
        print('Training and evaluating "{}" with num_reps={}.'.format(embedder_name, config.Eval.num_reps))
        embedder_job(embedder_name, params_df_row=params_df_row)
    #
    try:
        aggregation_job('matching')
        aggregation_job('identification')
    except RuntimeError:  # did not find any scores for identification
        pass
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reps', default=1, action='store', dest='reps', type=int,
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], required=False)
    parser.add_argument('-d', default=False, action='store_true', dest='debug', required=False)
    parser.add_argument('-l', default=False, action='store_true', dest='local', required=False)
    namespace = parser.parse_args()
    config.Eval.num_reps = namespace.reps
    if namespace.debug:
        config.Eval.debug = True
    if namespace.local:
        run_on_host(params_df)
    else:
        run_on_cluster()