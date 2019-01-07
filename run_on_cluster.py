from itertools import chain
import socket

from src.params import CountParams, RNNParams, Word2VecParams, RandomControlParams, GloveParams
from src.params import gen_all_param_combinations
from src.embedders.glove import GloveEmbedder
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder
from src.jobs import embed_and_evaluate

from ludwigcluster.client import Client


# TODO ludwigcluster saves a configs.csv file that is loaded -
# TODO the configs file (which wil be unique to each node) can simply have the name of the embedder to train

# cluster-specific



