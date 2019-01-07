import argparse

from ludwigcluster.client import Client
from src import config
from src.params import params_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--replications', action='store', dest='reps', type=int, required=True)
    parser.add_argument('-t', '--test', action='store_true', dest='test', required=False)
    namespace = parser.parse_args()
    client = Client(project_name='2StageNLP')
    client.submit(config.Dirs.src, params_df, reps=namespace.reps, test=namespace.test)

    # TODO upload venv along with code? (childeshub, for example is not uploaded)