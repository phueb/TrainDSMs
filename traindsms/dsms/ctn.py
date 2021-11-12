import math
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas

from traindsms.dsms.network import NetworkBaseClass
from networkx.drawing.nx_agraph import graphviz_layout
from missingadjunct.corpus import Corpus
from missingadjunct.utils import make_blank_sr_df

VERBOSE = False


class CTN(NetworkBaseClass):
    """
    Constituent tree network, generated by joining parsed trees by shared nodes(constituent)

    """

    def __init__(self, parsed_corpus):
        NetworkBaseClass.__init__(self, parsed_corpus)
        self.freq_threshold = 1
        self.diamond_list = []
        self.network_type = 'ctn'
        self.lexical_network = None

    #########################################################################################

    # functions used to transform a parsed corpus (nested lists or tuples) to a corpus of
    # constituent trees

    #########################################################################################

    # for a given nested list, return a copy in tuple data type

    def return_tuple(self,list):
        if self.flat(list) == 1:
            return tuple(list)
        else:
            return tuple(e if type(e) != list else self.return_tuple(e) for e in list)

    #########################################################################################

    # Judge if a list(tuple) is nested

    def flat(self, l):
        t = 1
        for item in l:
            if type(item) == list or type(item) == tuple:
                t = 0
                break
        return t

    #########################################################################################

    # Given a nested list, return the corresponding tree data structure: including the
    # node set and the edge set of the tree. where the nodes are all constituents in the
    # nested list, and the edges are subconstituent relations.\

    # the complete_tree function just makes sure that it works for the trivial cases
    # where the tree is just a word.

    def create_tree(self,x):
        if type(x) == str:
            return [], [x]
        else:
            if type(x) == list:
                x = self.return_tuple(x)
            edge_set = []
            node_set = []
            for item in x:
                edge_set.append((item, x))
                node_set.append(item)
            if self.flat(x) == 1:
                return edge_set, node_set
            else:
                for item in x:
                    if type(item) == tuple:
                        tem_tree = self.create_tree(item)
                        edge_set.extend(tem_tree[0])
                        node_set.extend(tem_tree[1])
                return edge_set, node_set

    def complete_tree(self,x):
        if type(x) == str:
            return [], [x]
        else:
            tree = self.create_tree(x)
            if type(x) == list:
                x = self.return_tuple(x)
            tree[1].append(x)
            tree1 = (tree[0], tree[1])
            return tree1

    #########################################################################################

    # create the networkx Digraph by joining the trees of the corpus.
    # the return consits of the edge set, the node set, and also the networkx graphical object of the network.

    def train(self):
        network_edge = []
        network_node = []
        count = 0
        epoch = 0
        start_time = time.time()
        for sentence in self.corpus:
            tree = self.complete_tree(sentence)
            if type(sentence) != str:
                self.diamond_list.append(tree)
            edge_set = tree[0]
            node_set = tree[1]
            for node in node_set:
                if type(node) == str and node not in self.badwords:
                    if node not in self.word_dict:
                        self.word_dict[node] = len(self.word_dict)
                        self.freq_dict[node] = 1
                    else:
                        self.freq_dict[node] = self.freq_dict[node] + 1

            network_edge.extend(edge_set)
            network_node.extend(node_set)
            count = count + 1
            if count >= 100:
                count = 0
                epoch = epoch + 100
                if VERBOSE:
                    print("{} sentences added to the network.".format(epoch))
        end_time = time.time()
        time_joining_trees = end_time - start_time
        self.network = nx.DiGraph()
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
            weighted_network_edge.append(edge + (math.log10(network_edge_dict[edge]+1),))
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
            print('Nodes in the network:')
            for node in self.node_list:
                print(self.node_list.index(node), node)
            print()
            print(len(self.word_dict),self.word_dict)
            print()
        final_freq_dict = {}
        for word in self.freq_dict:
            if self.freq_dict[word] >= self.freq_threshold:
                final_freq_dict[word] = self.freq_dict[word]
        self.freq_dict  = final_freq_dict
        if VERBOSE:
            print(len(final_freq_dict))
        self.word_list = [word for word in self.word_dict]
        self.get_adjacency_matrix()
        self.get_constituent_net()
        self.path_distance_dict = {node:{} for node in self.node_list}
        self.undirected_network = self.network.to_undirected()
        self.diameter = nx.algorithms.distance_measures.diameter(self.undirected_network)



    ###########################################################################################

    # get the neighborhood of a node
    # where a 'neighborhood' refers to all the trees containing the given node.

    def get_neighbor_node(self, node):
        neighborhood = [node]
        G = self.network
        if G.out_degree(node) == 0 or type(G.out_degree(node)) != int:
            neighborhood = set(neighborhood)
            return neighborhood
        else:
            count = 1
            while count > 0:
                for n in neighborhood:
                    if G.out_degree(n) > 0:
                        for m in G.successors(n):
                            neighborhood.append(m)
                            if G.out_degree(m) > 0:
                                count = count + 1
                        neighborhood.remove(n)
                        count = count - 1
            real_neighborhood = [1]
            for tree in neighborhood:
                subtree_node = self.complete_tree(tree)[1]
                real_neighborhood.extend(subtree_node)
            real_neighborhood = set(real_neighborhood)
            real_neighborhood.difference_update({1})
            return real_neighborhood

    ###########################################################################################

    # get the 'highest' node, which have 0 out degree (sentence or clause) in the network.

    ###########################################################################################

    # get the constituent distance between linked word pairs (with a edge)

    # for words next to each other in the constituent net, they have an edge if they are at least
    # co-occur in the same tree. Take all trees (sentences) that they co-occur, the constituent
    # distance betweeen word A and B is the average of all distances on the co-occur trees.

    # In a constituent net, the distance between any two words are defined as follow:
    # 1.If they are not connected, then infinity,
    # 2.If they have an edge, defined as above
    # 3.If they are connected, yet there is no edge between, then the distacne is the length of
    # the weighted shortest path between them, where the weight of an edge is the constituent distances
    # between the word pairs linked by the edge.

    def get_constituent_edge_weight(self):
        l = len(self.word_dict)

        weight_matrix = np.zeros((l, l), float)
        count_matrix = np.zeros((l, l), float)

        count = 0
        epoch = 0
        start_time = time.time()

        for tree_info in self.diamond_list:
            sent_node = tree_info[1]
            sent_edge = tree_info[0]
            sent_words = []
            tree = nx.Graph()
            tree.add_edges_from(sent_edge)
            for node in sent_node:
                if type(node) == str and node not in self.badwords:
                    sent_words.append(node)

            for word1 in sent_words:
                id1 = self.word_dict[word1]
                for word2 in sent_words:
                    id2 = self.word_dict[word2]
                    if id1 != id2:
                        weight_matrix[id1][id2] = weight_matrix[id1][id2] + .5 ** (
                                    nx.shortest_path_length(tree, word1, word2) - 1)
                        # count_matrix[id1][id2] = count_matrix[id1][id2] + 1

            count = count + 1
            if count >= 100:
                count = 0
                epoch = epoch + 100
                if VERBOSE:
                    print("{} weights added to the weight matrix.".format(epoch))

        end_time = time.time()
        time_get_w_matrix = end_time - start_time
        if VERBOSE:
            print()
            print('{} used to get the weight matrix.'.format(time_get_w_matrix))
            print()

        return weight_matrix, count_matrix

    ###########################################################################################

    # create the constituent-net, which is the lexical netowrks derived from the CTN
    # The nodes of the net are the words in STN, while
    # for constituent-net, 2 words are linked if and only if they co-appear in at least one constituent

    def get_constituent_net(self):
        self.lexical_network = nx.Graph()
        weight_matrix, count_matrix = self.get_constituent_edge_weight()

        l = len(self.freq_dict)
        freq_list = [word for word in self.freq_dict]

        count = 0
        epoch = 0
        start_time = time.time()

        weight_normalizer = weight_matrix.sum(0)
        # count_normalizer = count_matrix.sum(0)

        for k in range(len(self.word_dict)):
            if weight_normalizer[k] == 0:
                # count_normalizer[k] = count_normalizer[k] + 1
                weight_normalizer[k] = weight_normalizer[k] + 1

        for i in range(l - 1):
            id1 = self.word_dict[freq_list[i]]
            for j in range(i + 1, l):
                id2 = self.word_dict[freq_list[j]]
                w = weight_matrix[i][j] / (weight_normalizer[i] * weight_normalizer[j]) ** .5
                if w > 0:
                    self.lexical_network.add_edge(freq_list[i], freq_list[j])
                    count = count + 1
                    if count >= 100:
                        count = 0
                        epoch = epoch + 100
                        if VERBOSE:
                            print("{} edges added to the constituent net.".format(epoch))

        end_time = time.time()
        time_get_C_net = end_time - start_time
        if VERBOSE:
            print()
            print('{} used to get the constituent net.'.format(time_get_C_net))
            print()

    ###########################################################################################

    # showing the neighborhood of a given node

    def get_neighbor(self, node):
        G = self.network
        choice_neighbor = list(self.get_neighbor_node(node))
        choice_net = G.subgraph(choice_neighbor)
        return choice_net

    def show_neighbor(self, word):
        word_neighbor = self.get_neighbor(word)

        nx.draw(word_neighbor, with_labels=True)

    ###########################################################################################

    # ploting the STN,
    def plot_network(self):
        steven = self.network
        plt.title('constituent tree network', loc= 'center')
        pos = graphviz_layout(steven, prog='dot')
        edges = steven.edges()
        weights = [steven[u][v]['weight'] for u, v in edges]
        nx.draw(steven, pos, with_labels=True, width = weights)
        plt.show()





    ###########################################################################################

    # for every word pair, compute the constituent distance between the pair
    # the lengths of all paths between the word pair

    def compute_distance_matrix(self, word_list1, word_list2):
        G = self.lexical_network
        l1 = len(word_list1)
        l2 = len(word_list2)
        distance_matrix = np.zeros((l1, l2), float)

        count = 0
        epoch = 0
        for i in range(l1):
            for j in range(l2):
                pair = [word_list1[i], word_list2[j]]
                if nx.has_path(G, pair[0], pair[1]):
                    distance = nx.shortest_path_length(G, pair[0], pair[1])
                else:
                    distance = np.inf
                distance_matrix[i][j] = round(distance, 3)

                count = count + 1
                if count >= 5:
                    count = 0
                    epoch = epoch + 5
                    print("{} pairs of distance caculated".format(epoch))

        table = pandas.DataFrame(distance_matrix, word_list1, word_list2)
        return table, distance_matrix

    def compute_similarity_matrix(self, word_list, neighbor_size):
        G = self.network.to_undirected()
        l = len(word_list)
        graph_list = []
        for word in word_list:
            g = self.get_sized_neighbor(word, neighbor_size)
            graph_list.append(g)
            # print(word, g.nodes)
            # print()

        similarity_matrix = np.zeros((l, l), float)
        for i in range(l):
            for j in range(i, l):
                similarity_matrix[i][j] = round(1 / (1 + nx.graph_edit_distance(graph_list[i], graph_list[j])), 3)
                # print(word_list[i],word_list[j],similarity_matrix[i][j])

        table = pandas.DataFrame(similarity_matrix, word_list, word_list)
        return table, similarity_matrix


def test_model():

    test_corpus = Corpus(include_location=False,
                    include_location_specific_agents=False,
                    num_epochs=1000,
                    seed=1,
                    )
    trees = []
    for t in test_corpus.get_trees():  # a sentence is a tree
        trees.append(t)

    dsm = CTN(trees)
    dsm.train()



    df_blank = make_blank_sr_df()
    df_results = df_blank.copy()
    instruments = df_blank.columns[3:]
    assert set(instruments).issubset(test_corpus.vocab)




    # fill in blank data frame with semantic-relatedness scores
    for verb_phrase, row in df_blank.iterrows():
        verb, theme = verb_phrase.split()
        scores = []

        if (verb, theme) in dsm.node_list:
            sr_verb = dsm.activation_spreading_analysis(verb, dsm.node_list, avoid = [((verb,theme),theme)])
            sr_theme = dsm.activation_spreading_analysis(theme, dsm.node_list, avoid=[((verb, theme), verb)])
        else:
            sr_verb = dsm.activation_spreading_analysis(verb, dsm.node_list, avoid=[])
            sr_theme = dsm.activation_spreading_analysis(theme, dsm.node_list, avoid=[])

        for instrument in instruments:  # instrument columns start after the 3rd column
            sr = math.log(sr_verb[instrument] * sr_theme[instrument])
            scores.append(sr)

        # collect sr scores in new df
        df_results.loc[verb_phrase] = [row['verb-type'], row['theme-type'], row['phrase-type']] + scores

    print(df_results.loc['preserve pepper'].round(3))
    df_results.to_csv('df_test_ctn.csv')


test_model()

# This is an attempt to get access to activation-spreading distance for all paths. It failed as number of possible paths
# could be very large basing on the graph structure, and time complexity could be exponential
'''''''''
def test_activation():
    test_corpus = Corpus(include_location=False,
                         include_location_specific_agents=False,
                         num_epochs=1,
                         seed=1,
                         )
    trees = []
    for t in test_corpus.get_trees():  # a sentence is a tree
        trees.append(t)

    dsm = CTN(trees)
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



#test_activation()
'''''''''


