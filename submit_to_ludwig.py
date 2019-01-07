import argparse

from ludwigcluster.client import Client
from src import config
from src.params import params_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--upload_all', default=False, action='store_true', dest='upload_all', required=False)
    parser.add_argument('-t', '--test', action='store_true', dest='test', required=False)
    namespace = parser.parse_args()
    client = Client(project_name='2StageNLP')
    upload_dirs = ['src', 'corpora', 'categories', 'tasks'] if namespace.upload_all else ['src']
    client.submit(upload_ps=[config.Dirs.root / d for d in upload_dirs],
                  params_df=params_df,
                  reps=1,
                  test=namespace.test,
                  check_reps=False,
                  worker='lecun')