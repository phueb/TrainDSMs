import argparse

from src.jobs import embedder_job
from src.params import params_df


def run_on_cluster():
    """
    run multiple jobs on multiple LudwigCluster nodes.
    """
    embedder_names = params_df['embedder_name'].values
    for embedder_name in embedder_names:
        print('Found embedder_name={} in params_df.'.format(embedder_name))
        embedder_job(embedder_name)


def run_on_host(embedder_name):
    """
    run jobs on the local host for testing/development
    """
    print('Running {} job locally'.format(embedder_name))
    embedder_job(embedder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=None, action='store', dest='embedder_name', type=str, required=False)
    namespace = parser.parse_args()
    if namespace.embedder_name is not None:
        run_on_host(namespace.embedder_name)
    else:
        run_on_cluster()