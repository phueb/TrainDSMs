import math
import networkx as nx
import time

from missingadjunct.corpus import Corpus


from traindsms.dsms.lon import LON
from traindsms.params import LONParams


def main():
    """
    This is an attempt to get access to activation-spreading distance for all paths.
    It failed as number of possible paths
    could be very large basing on the graph structure, and time complexity could be exponential
    """

    corpus = Corpus(include_location=False,
                    include_location_specific_agents=False,
                    complete_epoch=True,
                    num_epochs=0,
                    seed=1,
                    )
    sentences = []
    for s in corpus.get_sentences():  # a sentence is a string
        tokens = s.split()
        sentences.append(tokens)

    params = LONParams()
    dsm = LON(params, sentences)
    dsm.train()

    diameter = nx.algorithms.distance_measures.diameter(dsm.undirected_network)
    print(diameter)

    for node in dsm.node_list:
        start_time = time.time()
        dsm.get_path_distance(node, 1, [])
        end_time = time.time()
        time_getting_paths = end_time - start_time
        print()
        print('{} used to get paths from {}'.format(time_getting_paths, node))
        print()
        node_paths = dsm.path_distance_dict[node]
        for target_node in node_paths:
            print(node, target_node)
            print(node_paths[target_node])

