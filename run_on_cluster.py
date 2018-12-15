from itertools import chain
import socket

from src.params import CountParams, RNNParams, Word2VecParams, RandomControlParams, GloveParams
from src.params import gen_all_param_combinations
from src.embedders.glove import GloveEmbedder
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder

from src.experiment import embed_and_evaluate

# cluster-specific  # TODO
hostname = socket.gethostname()
hostname2embedder_name = {'hinton': ['ww', 'wd'],
                          'hoff': ['sg', 'cbow'],
                          'hebb': ['srn', 'lstm'],
                          'norman': ['random_normal', 'random_uniform'],
                          'pitts': ['glove']}


embedders = chain(
    (CountEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(CountParams)),
    (GloveEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(GloveParams)),
    (RNNEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(RNNParams)),
    (RandomControlEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(RandomControlParams)),
    (W2VecEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(Word2VecParams)),
)

# run full experiment
while True:
    # get embedder
    try:
        embedder = next(embedders)
    except RuntimeError as e:
        print('//////// WARNING: embedder raised RuntimeError:')
        print(e)
        continue
    except StopIteration:
        print('Finished experiment')
        break
    # embed
    # TODO cluster-specific
    if embedder.name in hostname2embedder_name[hostname]:
        embed_and_evaluate(embedder)