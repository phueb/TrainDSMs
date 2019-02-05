import argparse


from ludwigcluster.client import Client
from ludwigcluster.config import SFTP
from ludwigcluster.utils import list_all_param2vals

from two_stage_nlp import config
from two_stage_nlp.params import CountParams, RNNParams, Word2VecParams, RandomControlParams
from two_stage_nlp.jobs import preprocessing_job


"""
Do not --skip-data if any code related to corpus has been modified. 
This ensures that no old corpus data is used by workers.
"""

if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reps', default=2, action='store', dest='reps', type=int,
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], required=False)
    parser.add_argument('-w', '--worker', default=None, action='store', dest='worker',
                        choices=SFTP.worker_names, required=False)
    parser.add_argument('-s', '--skip_data', default=False, action='store_true', dest='skip_data', required=False)
    parser.add_argument('-t', '--test', action='store_true', dest='test', required=False)
    parser.add_argument('-p', '--preprocess', action='store_true', default=False, dest='preprocess', required=False)
    namespace = parser.parse_args()
    # preprocess corpus + save corpus data to file server
    if namespace.preprocess:
        print('Preprocessing corpus...')
        deterministic_w2f, vocab, docs, numeric_docs = preprocessing_job()
        print()
    # create all possible hyperparameter configurations
    update_d = {'corpus_name': config.Corpus.name, 'num_vocab': config.Corpus.num_vocab}
    param2val_list = list_all_param2vals(RandomControlParams, update_d) + \
                     list_all_param2vals(CountParams, update_d) + \
                     list_all_param2vals(RNNParams, update_d) + \
                     list_all_param2vals(Word2VecParams, update_d)
    # submit
    data_dirs = ['tasks'] if not namespace.skip_data else []
    client = Client(config.Dirs.remote_root.name)
    client.submit(src_ps=[config.Dirs.src],
                  data_ps=[config.Dirs.root / d for d in data_dirs],
                  param2val_list=param2val_list,
                  reps=namespace.reps,
                  test=namespace.test,
                  worker=namespace.worker)
