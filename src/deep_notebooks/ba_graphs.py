import itertools

import numpy as np
import igraph

from deep_notebooks.ba_functions import get_correlation_matrix
from spn.structure.Base import Leaf


def get_graph(spn, context):
    color_dict = {"Pro": "blue", "Sum": "green"}
    feature_names = context.feature_names
    g = igraph.Graph()
    labels = []
    colors = []
    
    def recurse(node):
        label = feature_names[node.scope[0]] if isinstance(node, Leaf) \
            else node.name[:3]
        labels.append(label)
        colors.append(color_dict.get(label, "red"))
        if not isinstance(node, Leaf):
            for c in node.children:
                g.add_vertices([c.name])
                g.add_edges([(node.name, c.name)])
                recurse(c)
    g.add_vertices([spn.name])
    recurse(spn)
    colors[0] = "white"
    g.vs["label"] = labels
    g.vs["color"] = colors
    return g


def get_correlation_graph(spn, context):
    features = context.feature_names
    features = list(enumerate(features))
    combinations = list(itertools.combinations(features, 2))
    cor = get_correlation_matrix(spn)
    g = igraph.Graph()
    labels = []
    weights = []
    colors = []
    for f in features:
        g.add_vertex(f[0])
        labels.append(f[1])
    for c in combinations:
        weight = np.round(cor[c[0][0], c[1][0]], 2)
        if weight < 0:
            colors.append("red")
        else:
            colors.append("blue")
        if not np.isnan(weight):
            g.add_edge(c[0][0], c[1][0])
            weights.append(weight)
    g.vs["label"] = labels
    g.vs["color"] = ["white" for i in labels]
    g.es["weight"] = weights
    g.es["color"] = colors
    return g


def analyze_correlation_graph(spn):
    features = spn.featureNames
    features = list(enumerate(features))
    combinations = list(itertools.combinations(features, 2))
    cor = get_correlation_matrix(spn)
    g = igraph.Graph()
    labels = []
    weights = []
    for f in features:
        g.add_vertex(f[0])
        labels.append(f[1])
    for c in combinations:
        weight = np.round(cor[c[0][0], c[1][0]], 2)
        if not np.isnan(weight):
            g.add_edge(c[0][0], c[1][0])
            weights.append(weight)
    g.vs["label"] = labels
    g.vs["color"] = ["white" for i in labels]
    g.es["weight"] = weights
    return g


def get_predictive_graph(spn, instance, categorical):
    norm = instance.copy(deep=True)
    norm[:, categorical] = np.nan
    vertex_size = 20
    color_dict = {"Pro": "blue", "Sum": "green"}
    g = igraph.Graph()
    labels = []
    colors = []
    sizes = []
    
    def recurse(node):
        label = node.featureName if isinstance(node, Leaf) else node.name[:3]
        labels.append(label)
        colors.append(color_dict.get(label, "red"))
        sizes.append(vertex_size * (node.eval(instance) - node.eval(norm)))
        if not isinstance(node, Leaf):
            for c in node.children:
                g.add_vertices([c.name])
                g.add_edges([(node.name, c.name)])
                recurse(c)
    g.add_vertices([spn.name])
    recurse(spn)
    colors[0] = "white"
    g.vs["label"] = labels
    g.vs["color"] = colors
    g.vs["size"] = sizes
    return g

