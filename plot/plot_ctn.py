import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx


def plot_network(network):
    steven = network
    plt.title('constituent tree network', loc= 'center')
    pos = graphviz_layout(steven, prog='dot')
    edges = steven.edges()
    weights = [steven[u][v]['weight'] for u, v in edges]
    nx.draw(steven, pos, with_labels=True, width = weights)
    plt.show()
