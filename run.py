import argparse
import pickle
import socket

from two_process_nlp import config
from two_process_nlp.jobs import main_job, aggregation_job, preprocessing_job
from two_process_nlp.params import CountParams, RandomControlParams

hostname = socket.gethostname()


def run_on_cluster():
    """
    run multiple jobs on multiple LudwigCluster nodes.
    """
    config.LocalDirs.corpora = config.RemoteDirs.root / 'corpora'
    config.LocalDirs.tasks = config.RemoteDirs.root / 'tasks'
    #
    p = config.RemoteDirs.root / '{}_param2val_chunk.pkl'.format(hostname)
    with p.open('rb') as f:
        param2val_chunk = pickle.load(f)
    for param2val in param2val_chunk:
        try:
            main_job(param2val)
        except NotImplementedError as e:
            print(e)
    #
    aggregation_job(verbose=False)
    print('Finished {} jobs\n'.format(config.RemoteDirs.root.name))


def run_on_host(params):
    """
    run jobs on the local host for testing/development
    """
    from ludwigcluster.utils import list_all_param2vals
    #
    for param2val in list_all_param2vals(params, update_d={'param_name': 'test', 'job_name': 'test'}):
        try:
            main_job(param2val)
        except NotImplementedError as e:
            print(e)
    #
    print('Finished {} jobs\n'.format(config.RemoteDirs.root.name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', default=False, action='store_true', dest='local', required=False)
    parser.add_argument('-d', default=False, action='store_true', dest='debug', required=False)
    parser.add_argument('-p', default=False, action='store_true', dest='preprocess', required=False)
    parser.add_argument('-r', default=False, action='store_true', dest='random_normal', required=False)
    namespace = parser.parse_args()
    if namespace.debug:
        config.Eval.debug = True
        config.Eval.num_epochs_matching = 100
        config.Eval.num_epochs_identification = 100  # TODO
    if namespace.preprocess:
        preprocessing_job()
    if namespace.local:
        run_on_host(CountParams if not namespace.random_normal else RandomControlParams)
    else:
        run_on_cluster()