import matplotlib.pyplot as plt
import math
import networkx as nx


def plot_activation(graph, activation_recorder):
    """
    plot the lexical network, where all nodes are words
    """

    plt.title('lexical network with activation dispersion', loc= 'center')

    color_list = []
    for node in graph:
        color_list.append(math.log(activation_recorder[0, graph.node_list.index(node)]))

    vmin = min(color_list)
    vmax = max(color_list)
    cmap = plt.cm.cool
    nx.draw(graph, with_labels=True, node_color=color_list, cmap= cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm)
    plt.show()
