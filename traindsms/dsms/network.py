# This module define class Dnetwork (Distributional Networks)
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.sparse import lil_matrix

from nltk.corpus import stopwords
from nltk.corpus import wordnet

VERBOSE = False


class NetworkBaseClass:
    """
    abstract class inherited by CTN (constituent-tree network) and LON (liner-order network)
    """

    def __init__(self, corpus):
        self.corpus = corpus
        self.network_type = None
        self.freq_threshold = None # filter out word with low frequency
        self.network = None
        self.undirected_network = None
        self.diameter = None
        self.node_list = []
        self.word_list = []
        self.word_dict = {}
        self.freq_dict = {}
        self.pos_list = ['n', 'v', 'a', 'r', None]
        self.badwords = set(stopwords.words('english'))
        self.lexical_network = None
        self.adjacency_matrix = None
        self.path_distance_dict = {node:{} for node in self.node_list}# activation distances for all paths in the graph
        # keys are nodes, values are distances from the key node to all other nodes in the graph.

    ###########################################################################################
    # corpus pre-processing
    ###########################################################################################

    # get all open class words in the network
    # sort the words by default pos tags.

    def get_pos_tag(self):
        pos_dict = {}
        for tag in self.pos_list:
            pos_dict[tag] = []

        for word in self.word_list:
            pos_tag = self.get_wordnet_tag(word)
            pos_dict[pos_tag].append(word)

        return pos_dict

    # get the wordnet pos tag for a word(a string, type:str)

    def get_wordnet_tag(self, word):
        nltk_tag = nltk.pos_tag([word])[0][1]
        wordnet_tag = self.nltk_tag_to_wordnet_tag(nltk_tag)
        return wordnet_tag

    # transform an nltk POS tag to a wordnet POS tag

    def nltk_tag_to_wordnet_tag(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    ###########################################################################################
    # general graph functions
    ###########################################################################################


    # This is a function that returns a neighbor (consisting of nodes) of a specified node with an assigned size.
    # For example, given a graph G, and a node u in G, and let size be 2, it returns all nodes in G which are up to
    # 2 edges away from u.

    def get_sized_neighbor_node(self, Gragh, node, size):
        i = 0
        neighbor_node_dict = {node: 0}
        neighbor_node_list = [node]

        while i < size:
            for node in neighbor_node_dict:
                if neighbor_node_dict[node] == i:
                    for neighbor_node in Gragh[node]:
                        neighbor_node_list.append(neighbor_node)
            for node in neighbor_node_list:
                if node not in neighbor_node_dict:
                    neighbor_node_dict[node] = i + 1
            i = i + 1

        return neighbor_node_list

    # showing the sized neighborhood of a given node

    def get_sized_neighbor(self, node, size):
        G = self.network[2].to_undirected()
        choice_neighbor = self.get_sized_neighbor_node(G, node, size)
        choice_net = G.subgraph(choice_neighbor)
        return choice_net

    def show_sized_neighbor(self, word, size):
        word_neighbor = self.get_sized_neighbor(word, size)

        nx.draw(word_neighbor, with_labels=True)

    ###########################################################################################
    # spreading activation measure of semantic relatedness on network models (forming networks)
    ###########################################################################################

    def get_adjacency_matrix(self):
        W = nx.adjacency_matrix(self.network, nodelist=self.node_list)
        W.todense()
        length = W.shape[0]
        W = W + np.transpose(W)
        W = lil_matrix(W)

        normalizer = W.sum(1)
        for i in range(length):
            for j in range(length):
                if normalizer[i][0, 0] == 0:
                    W[i, j] = 0
                else:
                    W[i, j] = W[i, j] / normalizer[i][0, 0]

        self.adjacency_matrix = W



    def activation_spreading_analysis(self, source, target, avoid):
        # Spreading activation to measure the functional distance from source to target
        # where source is one item, target is a list(of items), avoid is a list of directed edges (in the form of:(a,b)
        # indicating the edge from a to b) nodes that source need to avoid, in other words, not allowing spreading
        # activation from the source pass through the edge
        # returns a sr_dictionary consisting of sr from the source to all targets

        W = self.adjacency_matrix.copy()
        length = W.shape[0]
        #print('Measuring semantic relatedness from '+ source)

        activation = np.zeros((1, length), float)
        fired = np.ones((1, length), float)
        activation[0, self.node_list.index(source)] = 1  # source is activated
        fired[0, self.node_list.index(source)] = 0  # source has fired
        for edge in avoid:
            from_id = self.node_list.index(edge[0])
            to_id = self.node_list.index(edge[1])
            W[from_id, to_id] = 0
            fired[0, to_id] = 0 # avoided node is considered as fired
        activation_recorder = activation
        last_fired = np.zeros((1, length), float)

        while fired.any() or last_fired.any():
            activation = activation * W
            activation_recorder = activation_recorder + np.multiply(fired, activation)+ \
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
        node_dict = {}
        sorted_dict = {}

        '''''''''
        if dg == 'constituent':
            for node in node_list:
                node_dict[node] = activation_recorder[0,node_list.index(node)]
                sorted_dict = {k: v for k, v in sorted(node_dict.items(), key=lambda item: item[1], reverse=True)}
            for node in sorted_dict:
                print((node, sorted_dict[node]))
        '''''''''

        semantic_relatedness_dict = {}
        for word in target:
            semantic_relatedness_dict[word] = activation_recorder[0, self.node_list.index(word)]

        return semantic_relatedness_dict



    def get_path_distance(self, source, source_activation, visited):
        # for a given node (source) get the activation-spreading distance to all other nodes in the graph through all
        # paths


        if len(visited) == 0:
            print('Getting activation from ' + str(source))

        visited.append(source)
        #print(len(visited), visited)
        original_source = visited[0]
        if source not in self.path_distance_dict[original_source]:
            self.path_distance_dict[original_source][source]={tuple(visited) : source_activation} # create the dict for
            # all paths from the original source to the node focusing on
        else:
            self.path_distance_dict[original_source][source][tuple(visited)] = source_activation
        if len(visited) < 2*self.diameter:
            for node in self.undirected_network.neighbors(source):
                if node not in visited:
                    node_activation = source_activation * \
                                      self.adjacency_matrix[self.node_list.index(source), self.node_list.index(node)]
                    self.get_path_distance(node, node_activation, visited.copy())



    # plot the lexical network, where all nodes are words

    def plot_activation(self, activation_recorder):
        if self.network_type == 'Co-occurrence':
            lexical_net = self.network
        else:
            lexical_net = self.lexical_network
        plt.title('lexical network with activation dispersion on' + self.network_type, loc= 'center')

        color_list = []
        for node in lexical_net:
            color_list.append(math.log(activation_recorder[0, self.node_list.index(node)]))

        vmin = min(color_list)
        vmax = max(color_list)
        cmap = plt.cm.cool
        nx.draw(lexical_net, with_labels=True, node_color=color_list, cmap= cmap)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        plt.colorbar(sm)
        plt.show()

'''''''''
visitedList = [[]]

def depthFirst(graph, currentVertex, visited):
    visited.append(currentVertex)
    for vertex in graph[currentVertex]:
        if vertex not in visited:
            depthFirst(graph, vertex, visited.copy())
    visitedList.append(visited)

depthFirst(graph, 0, [])

print(visitedList)
'''''''''
