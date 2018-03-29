'''
Created on March 29, 2018

@author: Alejandro Molina
'''


def plot_spn2(spn, fname="plot.pdf"):
    import networkx as nx
    from networkx.drawing.nx_pydot import graphviz_layout

    import matplotlib.pyplot as plt
    from spn.structure.Base import Sum, Product, Leaf, get_nodes_by_type
    import numpy as np

    all_nodes = get_nodes_by_type(spn)

    g = nx.DiGraph()

    labels = {}
    edge_labels = {}
    for n in all_nodes:

        if isinstance(n, Sum):
            label = "+"
        elif isinstance(n, Product):
            label = "x"
        else:
            label = "V" + str(n.scope[0])
        g.add_node(n.id)
        labels[n.id] = label

        if isinstance(n, Leaf):
            continue
        for i, c in enumerate(n.children):
            edge_label = ""
            if isinstance(n, Sum):
                edge_label = np.round(n.weights[i],2)
            g.add_edge(c.id, n.id, weight=edge_label)


    pos = graphviz_layout(g, prog='dot', args="height=200")
    #pos = nx.drawing.layout.rescale_layout(pos, 10)
    plt.figure(figsize=(18, 12))
    ax = plt.gca()
    ax.invert_yaxis()

    nx.draw(g, pos, with_labels=True, arrows=False, node_color='#DDDDDD', edge_color='#DDDDDD',width=1, node_size=250, labels=labels, font_size=6)
    ax.collections[0].set_edgecolor("#888888")
    edge_labels = nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=nx.get_edge_attributes(g,'weight'), font_size=5, clip_on=False, alpha=0.6)
    plt.tight_layout()
    plt.savefig(fname)
