import argparse
import pickle

from src.jobs import preprocessing_job
from ludwigcluster.client import Client
from ludwigcluster.config import SFTP
from src import config
from src.params import CountParams, RNNParams, Word2VecParams, RandomControlParams
from src.params import gen_combinations

"""
Do not --skip-data if any code related to corpus has been modified. 
This ensures that no old corpus data is used by workers.
This also requires deleting of any old w2freq or vocab txt files in the corpora folder.
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
    # preprocess
    if namespace.preprocess:
        print('Preprocessing corpus...')
        deterministic_w2f, vocab, docs, numeric_docs = preprocessing_job()
        # save w2freq
        p = config.Dirs.remote_root / '{}_w2freq.txt'.format(config.Corpus.name)
        with p.open('w') as f:
            for probe, freq in deterministic_w2f.items():
                f.write('{} {}\n'.format(probe, freq))
        # save vocab
        p = config.Dirs.remote_root / '{}_{}_vocab.txt'.format(config.Corpus.name, config.Corpus.num_vocab)
        with p.open('w') as f:
            for v in vocab:
                f.write('{}\n'.format(v))
        # save numeric_docs
        p = config.Dirs.remote_root / '{}_{}_numeric_docs.pkl'.format(config.Corpus.name, config.Corpus.num_vocab)
        with p.open('wb') as f:
            pickle.dump(numeric_docs, f)
        # save docs
        p = config.Dirs.remote_root / '{}_{}_docs.pkl'.format(config.Corpus.name, config.Corpus.num_vocab)
        with p.open('wb') as f:
            pickle.dump(docs, f)
        print()
    # param2val - reps are added by ludwigcluster
    param2val_list = list(gen_combinations(CountParams)) + \
                     list(gen_combinations(RNNParams)) + \
                     list(gen_combinations(Word2VecParams)) + \
                     list(gen_combinations(RandomControlParams))
    # submit
    data_dirs = ['corpora', 'tasks'] if not namespace.skip_data else []  # TODO upload tasks to server above, and don't let ludwigcluster upload any data to workers, only code
    client = Client(config.Dirs.remote_root.name, delete_delta=24)
    client.submit(src_ps=[config.Dirs.src],
                  data_ps=[config.Dirs.root / d for d in data_dirs],  # TODO don't need this? (all data is saved to server not worker)
                  param2val_list=param2val_list,
                  reps=namespace.reps,
                  test=namespace.test,
                  worker=namespace.worker)
