import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix
from typing import List, Dict

VERBOSE = False


class NetworkBaseClass:
    """
    abstract class inherited by CTN (constituent-tree network) and LON (liner-order network)
    """

    def __init__(self):
        self.network = None
        self.undirected_network = None
        self.diameter = None
        self.node_list = []
        self.word_dict = {}
        self.adjacency_matrix = None
        self.path_distance_dict = {node: {} for node in self.node_list}
        # keys are nodes, values are distances from the key node to all other nodes in the graph.

    def get_sized_neighbor_node(self, graph, node, size):
        """
        return all nodes in G which are up some number of edges away from a given node
        """
        i = 0
        neighbor_node_dict = {node: 0}
        neighbor_node_list = [node]

        while i < size:
            for node in neighbor_node_dict:
                if neighbor_node_dict[node] == i:
                    for neighbor_node in graph[node]:
                        neighbor_node_list.append(neighbor_node)
            for node in neighbor_node_list:
                if node not in neighbor_node_dict:
                    neighbor_node_dict[node] = i + 1
            i = i + 1

        return neighbor_node_list

    def get_sized_neighbor(self, node, size):
        """
        showing the sized neighborhood of a given node
        """
        net = self.network[2].to_undirected()
        choice_neighbor = self.get_sized_neighbor_node(net, node, size)
        choice_net = net.subgraph(choice_neighbor)
        return choice_net

    def get_adjacency_matrix(self):
        adj_mat = nx.adjacency_matrix(self.network, nodelist=self.node_list)
        adj_mat.todense()
        length = adj_mat.shape[0]
        adj_mat = adj_mat + np.transpose(adj_mat)
        adj_mat = lil_matrix(adj_mat)

        normalizer = adj_mat.sum(1)
        for i in range(length):
            for j in range(length):
                if normalizer[i][0, 0] == 0:
                    adj_mat[i, j] = 0
                else:
                    adj_mat[i, j] = adj_mat[i, j] / normalizer[i][0, 0]

        self.adjacency_matrix = adj_mat

    def activation_spreading_analysis(self,
                                      source: str,
                                      targets: List[str],
                                      excluded_edges,  # a list of directed edges (e.g.(a,b)) that are excluded
                                      ) -> Dict[str, float]:
        """
        a spreading-activation measure of the "semantic relatedness" (sr) from source to targets.

        return a sr_dictionary consisting of sr from the source to all targets
        """

        adj_mat = self.adjacency_matrix.copy()
        length = adj_mat.shape[0]

        activation = np.zeros((1, length), float)
        fired = np.ones((1, length), float)
        activation[0, self.node_list.index(source)] = 1  # source is activated
        fired[0, self.node_list.index(source)] = 0  # source has fired
        for edge in excluded_edges:
            from_id = self.node_list.index(edge[0])
            to_id = self.node_list.index(edge[1])
            adj_mat[from_id, to_id] = 0
            fired[0, to_id] = 0 # avoided node is considered as fired
        activation_recorder = activation
        last_fired = np.zeros((1, length), float)

        while fired.any() or last_fired.any():
            activation = activation * adj_mat
            activation_recorder = activation_recorder + np.multiply(fired, activation) + \
                                  np.multiply(last_fired,activation)
            # record the first time arrived
            # activation, which stands for the semantic relatedness from the source to the activated node
            last_fired = np.zeros((1,length),float)
            for i in range(length):
                # a node which has not fired get activated, automatically updated to fired
                if fired[0, i] == 1 and activation[0, i] != 0:
                    fired[0, i] = 0
                    last_fired[0, i] = 1

        sorted_activation = activation_recorder.tolist()[0]
        sorted_activation.sort(reverse=True)

        # node_dict = {}
        # sorted_dict = {}
        # if dg == 'constituent':
        #     for node in node_list:
        #         node_dict[node] = activation_recorder[0,node_list.index(node)]
        #         sorted_dict = {k: v for k, v in sorted(node_dict.items(), key=lambda item: item[1], reverse=True)}
        #     for node in sorted_dict:
        #         print((node, sorted_dict[node]))

        semantic_relatedness_dict = {}
        for word in targets:
            semantic_relatedness_dict[word] = activation_recorder[0, self.node_list.index(word)]

        return semantic_relatedness_dict

    def get_path_distance(self, source, source_activation, visited):
        """
        for a given node, get the activation-spreading distance to all other nodes in the graph through all paths
        """

        if len(visited) == 0:
            print('Getting activation from ' + str(source))

        visited.append(source)

        original_source = visited[0]
        if source not in self.path_distance_dict[original_source]:
            # create the dict for all paths from the original source to the node focusing on
            self.path_distance_dict[original_source][source] = {tuple(visited): source_activation}
        else:
            self.path_distance_dict[original_source][source][tuple(visited)] = source_activation

        if len(visited) < 2*self.diameter:
            for node in self.undirected_network.neighbors(source):
                if node not in visited:
                    node_activation = source_activation * \
                                      self.adjacency_matrix[self.node_list.index(source), self.node_list.index(node)]
                    self.get_path_distance(node, node_activation, visited.copy())
