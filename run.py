import argparse
import pickle
import socket

from two_stage_nlp import config
from two_stage_nlp.jobs import two_stage_job, aggregation_job, preprocessing_job
from two_stage_nlp.params import CountParams, RandomControlParams, RNNParams

hostname = socket.gethostname()


def run_on_cluster():
    """
    run multiple jobs on multiple LudwigCluster nodes.
    """
    config.Dirs.corpora = config.Dirs.remote_root / 'corpora'
    config.Dirs.tasks = config.Dirs.remote_root / 'tasks'
    #
    p = config.Dirs.remote_root / '{}_param2val_chunk.pkl'.format(hostname)
    with p.open('rb') as f:
        param2val_chunk = pickle.load(f)
    for param2val in param2val_chunk:
        try:
            two_stage_job(param2val)
        except NotImplementedError as e:
            print(e)
    #
    aggregation_job(verbose=False)
    print('Finished {} jobs\n'.format(config.Dirs.remote_root.name))


def run_on_host():
    """
    run jobs on the local host for testing/development
    """
    from ludwigcluster.utils import list_all_param2vals
    #
    preprocessing_job()

    for param2val in list_all_param2vals(RandomControlParams, update_d={'param_name': 'test', 'job_name': 'test'}):
    # for param2val in list_all_param2vals(CountParams, update_d={'param_name': 'test', 'job_name': 'test'}):
        try:
            two_stage_job(param2val)
        except NotImplementedError as e:
            print(e)

        print('Finished {} jobs\n'.format(config.Dirs.remote_root.name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', default=False, action='store_true', dest='local', required=False)
    parser.add_argument('-d', default=False, action='store_true', dest='debug', required=False)
    namespace = parser.parse_args()
    if namespace.debug:
        config.Eval.debug = True
    if namespace.local:
        run_on_host()
    else:
        run_on_cluster()