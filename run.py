import argparse
import yaml

from src import config
from src.jobs import embedder_job, aggregation_job


def run_on_cluster():
    """
    run multiple jobs on multiple LudwigCluster nodes.
    """
    config.Dirs.corpora = config.Dirs.remote_root / 'corpora'
    config.Dirs.tasks = config.Dirs.remote_root / 'tasks'
    #
    for p in config.Dirs.param2val:
        with p.open('r') as f:
            param2val = yaml.load(f)
        embedder_job(param2val)
    #
    try:
        aggregation_job('matching')
        aggregation_job('identification')
    except RuntimeError:  # did not find any scores for identification
        pass
        print('Finished embedding + evaluation + aggregation.')
        print()


def run_on_host():  # TODO how to do reps locally?
    """
    run jobs on the local host for testing/development
    """
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default=False, action='store_true', dest='debug', required=False)
    parser.add_argument('-l', default=False, action='store_true', dest='local', required=False)
    namespace = parser.parse_args()
    if namespace.debug:
        config.Eval.debug = True
    if namespace.local:
        run_on_host()
    else:
        run_on_cluster()