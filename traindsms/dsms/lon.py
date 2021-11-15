import math
import time
import networkx as nx
import networkx as nx
from typing import List, Tuple

from traindsms.dsms.network import NetworkBaseClass
from traindsms.params import LONParams

VERBOSE = False


class LON(NetworkBaseClass):
    """
    co-occurrence graph, generated by joining linear sentence sequence by shared words
    """

    def __init__(self, linear_corpus):
        NetworkBaseClass.__init__(self, linear_corpus)
        self.freq_threshold = 1
        self.co_mat = None
        self.network_type = 'Co-occurrence'


    def train(self):
        network_edge = []
        network_node = []
        count = 0
        epoch = 0
        start_time = time.time()

        for sentence in self.corpus:
            node_set = sentence
            edge_set = []
            l = len(sentence)
            for i in range(l-1):
                edge_set.append((node_set[i],node_set[i+1]))
            for node in node_set:
                if node not in self.word_dict:
                    self.word_dict[node] = len(self.word_dict)
                    self.freq_dict[node] = 1
                else:
                    self.freq_dict[node] = self.freq_dict[node] + 1
            network_edge.extend(edge_set)
            network_node.extend(node_set)
            count = count + 1
            if count >= 1000:
                count = 0
                epoch = epoch + 1000
                if VERBOSE:
                    print("{} sentences added to the linear graph.".format(epoch))
        end_time = time.time()
        time_joining_trees = end_time - start_time
        self.network = nx.Graph()
        network_node_set = set(network_node)
        self.node_list = list(network_node_set)
        network_edge_dict = {}
        for edge in network_edge:
            if edge in network_edge_dict:
                network_edge_dict[edge] = network_edge_dict[edge] + 1
            else:
                network_edge_dict[edge] = 1
        weighted_network_edge = []
        for edge in network_edge_dict:
            weighted_network_edge.append(edge + (math.log10(network_edge_dict[edge] + 1),))
        if VERBOSE:
            print()
            print('Weighted Edges:')
            for edge in weighted_network_edge:
                print(self.node_list.index(edge[0]), self.node_list.index(edge[1]), edge)
            print()
        self.network.add_weighted_edges_from(weighted_network_edge)
        if VERBOSE:
            print()
            print('{} used to join the trees.'.format(time_joining_trees))
            print()
        final_freq_dict = {}
        for word in self.freq_dict:
            if self.freq_dict[word] >= self.freq_threshold:
                final_freq_dict[word] = self.freq_dict[word]
        self.freq_dict = final_freq_dict
        if VERBOSE:
            print(len(final_freq_dict))
        self.word_list = [word for word in self.word_dict]
        self.get_adjacency_matrix()
        self.path_distance_dict = {node: {} for node in self.node_list}
        self.undirected_network = self.network.to_undirected()
        self.diameter = nx.algorithms.distance_measures.diameter(self.undirected_network)

        return network_node_list, linear_doug, word_dict, final_freq_dict