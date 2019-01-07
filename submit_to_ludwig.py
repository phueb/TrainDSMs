import argparse

from ludwigcluster.client import Client
from src import config
from src.params import params_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--upload_data', default=False, action='store_true', dest='upload_data', required=False)
    parser.add_argument('-t', '--test', action='store_true', dest='test', required=False)
    namespace = parser.parse_args()
    data_dirs = ['corpora', 'categories', 'tasks'] if namespace.upload_data else []
    client = Client(config.Ludwig.project_name)
    client.submit(src_ps=[config.Dirs.src],
                  data_ps=[config.Dirs.root / d for d in data_dirs],
                  params_df=params_df,
                  reps=1,
                  test=namespace.test,
                  use_log=False,
                  worker='lecun')