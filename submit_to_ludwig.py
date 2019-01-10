import argparse

from ludwigcluster.client import Client
from ludwigcluster.config import SFTP
from src import config
from src.params import params_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reps', default=1, action='store', dest='reps', type=int,
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], required=False)
    parser.add_argument('-w', '--worker', default=None, action='store', dest='worker',
                        choices=SFTP.worker_names, required=False)
    parser.add_argument('-a', '--upload_data', default=False, action='store_true', dest='upload_data', required=False)
    parser.add_argument('-t', '--test', action='store_true', dest='test', required=False)
    namespace = parser.parse_args()
    config.Eval.num_reps = namespace.reps
    data_dirs = ['corpora', 'categories', 'tasks'] if namespace.upload_data else []
    client = Client(config.Ludwig.project_name)
    client.submit(src_ps=[config.Dirs.src],
                  data_ps=[config.Dirs.root / d for d in data_dirs],
                  params_df=params_df,
                  reps=1,
                  test=namespace.test,
                  use_log=False,
                  worker=namespace.worker)