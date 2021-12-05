import math
import networkx as nx
import time

from missingadjunct.corpus import Corpus

from traindsms.dsms.ctn import CTN
from traindsms.params import CTNParams


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
    seq_parsed = []
    for t in corpus.get_trees():  # a tree is a nested tuple
        seq_parsed.append(t)

    params = CTNParams(excluded_tokens=None)
    dsm = CTN(params, corpus.token2id, seq_parsed)
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
        # node_paths = dsm.path_distance_dict[node]
        # for target_node in node_paths:
        # print(node, target_node)
        # print(node_paths[target_node])

if __name__ == '__main__':
    main()