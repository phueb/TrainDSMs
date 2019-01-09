import argparse
from pathlib import Path

from src import config
from src.jobs import embedder_job
from src.params import params_df


def run_on_cluster():
    """
    run multiple jobs on multiple LudwigCluster nodes.
    """
    for n, params_df_row in params_df.iterrows():
        embedder_class = params_df_row['embedder_class']
        print('Found embedder_class={} in params_df.'.format(embedder_class))
        # overwrite directories where data is stored and saved
        config.Dirs.runs = Path(params_df_row['runs_dir'])
        config.Dirs.corpora = Path(params_df_row['runs_dir'].replace('runs', 'corpora'))
        config.Dirs.tasks = Path(params_df_row['runs_dir'].replace('runs', 'tasks'))
        config.Dirs.categories = Path(params_df_row['runs_dir'].replace('runs', 'categories'))
        embedder_job(embedder_class)


def run_on_host(embedder_classes):
    """
    run jobs on the local host for testing/development
    """
    for embedder_class in embedder_classes:
        embedder_job(embedder_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    embedder_classes = ['w2vec', 'rnn', 'count', 'random_control']
    parser.add_argument('-n', default=None, action='store', dest='embedder_class',
                        type=str, required=False, choices=embedder_classes)
    parser.add_argument('-a', default=None, action='store_true', dest='run_all', required=False)
    parser.add_argument('-d', default=False, action='store_true', dest='debug', required=False)
    namespace = parser.parse_args()
    if namespace.debug:
        config.Eval.debug = True
    if namespace.embedder_class is not None:
        run_on_host([namespace.embedder_class])
    elif namespace.run_all:
        print('Running full experiment on local machine.')
        run_on_host(embedder_classes)
    else:
        run_on_cluster()