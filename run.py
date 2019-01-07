from itertools import chain
import argparse
import socket

from src import config
from src.params import CountParams, RNNParams, Word2VecParams, RandomControlParams, GloveParams
from src.params import gen_all_param_combinations
from src.embedders.glove import GloveEmbedder
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder

from src.jobs import embed_and_evaluate


def do_jobs(embedder_names):
    embedders = chain(
        (W2VecEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(Word2VecParams)),
        (RNNEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(RNNParams)),
        (CountEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(CountParams)),
        (GloveEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(GloveParams)),
        (RandomControlEmbedder(param2id, param2val) for param2id, param2val in
         gen_all_param_combinations(RandomControlParams)),
    )
    runtime_errors = []
    while True:
        try:
            embedder = next(embedders)
        except RuntimeError as e:
            print('WARNING: embedder raised RuntimeError:')
            print(e)
            runtime_errors.append(e)
            continue
        except StopIteration:
            print('Finished experiment')
            for e in runtime_errors:
                print('with RunTimeError:')
                print(e)
            break
        if embedder.name in embedder_names:
            embed_and_evaluate(embedder)


def run_on_cluster():
    """
    run multiple jobs on multiple LudwigCluster nodes.
    must be able to load multiple params and train a model for each
    """
    hostname = socket.gethostname()
    embedder_names = config.Ludwig.hostname2embedder_names[hostname]
    do_jobs(embedder_names)


def run_on_host(embedder_name):
    """
    run jobs on the local host for testing/development
    """
    print('Running {} job locally'.format(embedder_name))
    do_jobs([embedder_name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--n', default=None, action='store', dest='embedder_name', type=str, required=False)
    namespace = parser.parse_args()
    if namespace.embedder_name is not None:
        run_on_host(namespace.embedder_name)
    else:
        run_on_cluster()