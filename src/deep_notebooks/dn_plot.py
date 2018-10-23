import igraph
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.tools as tls
import plotly.plotly as py
from plotly.graph_objs import Heatmap, Layout, Scatter, Bar, Table, Histogram, ColorBar

import deep_notebooks.ba_graphs as g

from spn.algorithms.Marginalization import marginalize
from spn.structure.StatisticalTypes import Type
from spn.algorithms.Inference import likelihood, func_from_spn


def plot_marginal(spn, feature_id, dictionary=None, fname=None, detail=100):
    context = dictionary['context']
    scope = set([feature_id])
    domain = context.get_domains_by_scope(scope)[0]

    # marg = marginalize(spn, scope)
    size = len(spn.scope)

    is_categorical = feature_id in context.get_categoricals()

    if is_categorical:
        enc = dictionary['features'][feature_id]['encoder'].inverse_transform
        x_range = domain
        y_range = func_from_spn(spn, feature_id)(x_range).reshape(-1)
        data = [Bar(
                    x=enc([int(x) for x in x_range]),
                    y=y_range,
                    )
                ]
    else:
        domain = context.get_domains_by_scope(scope)[0]
        _min = domain[0]
        _max = domain[-1]

        x_range = np.linspace(_min, _max, detail)
        values = func_from_spn(spn, feature_id)(x_range).reshape(-1)
        data = [Scatter(
                    x=x_range,
                    y=values,
                    mode='lines',
                    )
                ]
    layout = dict(width=450,
                  height=450,
                  xaxis=dict(title=context.feature_names[feature_id]),
                  yaxis=dict(title='Probability density' if not is_categorical else 'Probability')
                  )

    if fname is None:
        return {'data': data, 'layout': layout}
    else:
        raise NotImplementedError


def plot_related_features(spn, featureId_x, featureId_y, detail=100, dictionary=None, evidence=None, fname=None):
    """
    Plots a 2d representation of the joint marginal probability of these two
    features.

    :param spn: the root node of the spn
    :param featureId_x: featureid of the first feature
    :param featureId_y: featureid of the second feature
    :param detail: granularity of the plotting grid
    :param dictionary: the data dictionary to extract meta information
    :param evidence: evidence to condition the plot on
    :param fname: file name to save the resulting plot
    :return: a plotly dictionary containing data and context
    """

    # construct the grid
    num_features = len(spn.scope)
    context = dictionary['context']
    categoricals = context.get_categoricals()
    domain_x = context.get_domains_by_scope([featureId_x])[0]
    domain_y = context.get_domains_by_scope([featureId_y])[0]
    feature_names = context.feature_names
    x_range = (domain_x[0],
               domain_x[-1])
    y_range = (domain_y[0],
               domain_y[-1])
    x_detail = detail
    y_detail = detail
    x_cat = False
    y_cat = False
    if featureId_x in categoricals:
        x_cat = True
        x_detail = len(domain_x)
        enc = dictionary['features'][featureId_x]['encoder'].inverse_transform
        x_range = domain_x
        print([int(x) for x in x_range])
        x_names = enc(x_range)
    if featureId_y in categoricals:
        y_cat = True
        y_detail = len(domain_y)
        enc = dictionary['features'][featureId_y]['encoder'].inverse_transform
        y_range = domain_y
        print(y_range)
        y_names = enc([int(y) for y in y_range])
    grid = np.mgrid[x_range[0]:x_range[-1]:x_detail*1j, y_range[0]:y_range[-1]:y_detail*1j]
    grid = grid.reshape(2,-1).T

    # construct query
    query = np.zeros((1,num_features))
    query[:] = np.nan
    query = np.repeat(query, grid.shape[0], axis=0)
    query[:, featureId_x] = grid[:, 0]
    query[:, featureId_y] = grid[:, 1]

    # calculate the probability and shape the array
    result = likelihood(spn, query)

    result.shape = (x_detail, y_detail)
    
    # plot
    data = [Heatmap(z=result,
            x=np.linspace(domain_y[0], domain_y[-1], y_detail) if not y_cat else y_names,
            y=np.linspace(domain_x[0], domain_x[-1], x_detail) if not x_cat else x_names,
            colorbar=ColorBar(
                title='Colorbar'
            ),
            colorscale='Hot')]
    layout = dict(width=450, 
                  height=450,
                  xaxis=dict(title=feature_names[featureId_y], autotick=True),
                  yaxis=dict(title=feature_names[featureId_x], autotick=True)
                 )

    if fname is None:
        return {'data': data, 'layout': layout}
    else:
        raise NotImplementedError
























def plot_related_features_nodes(spn, featureId_x, featureId_y, evidence=None, sample_size=10000, fname=None):
    root = spn.root
    for i, child in enumerate(spn.root.children):
        spn.root = child
        plot_related_features(spn, featureId_x, featureId_y, evidence=evidence, fname=str(i)+fname)
    spn.root = root


def plot_graph(graph=None, spn=None, fname=None, context=None):
    if not graph and not spn:
        raise ValueError
    elif not graph:
        graph = g.get_graph(spn, context)

    layout = graph.layout_kamada_kawai()
    edge_width = graph.es["weight"] if graph.is_weighted() else [1] * len(graph.es)
    size = 200 * int(np.sqrt(len(graph.vs)))

    visual_style = {}
    visual_style["vertex_size"] = 20
    visual_style["vertex_label"] = graph.vs["label"]
    if graph.is_weighted():
        visual_style["edge_width"] = [abs(x)*5 for x in edge_width]
    visual_style["edge_label"] = [x if graph.is_weighted() else "" for x in edge_width]
    visual_style["layout"] = layout
    visual_style["bbox"] = (size, size)
    visual_style["margin"] = 100
    visual_style["vertex_label_dist"] = 3
    visual_style["vertex_label_size"] = 20
    visual_style["vertex_label_angle"] = 1

    if fname:
        igraph.plot(graph, fname, **visual_style)
    return graph, visual_style


def plot_conditional(spn, featureId, evidence, fname=None, detail=100):
    marg_ids = [featureId]
    if evidence is not None:
        query_ids = np.isfinite(evidence)
        query_ids[:,featureId] = True
        marg_ids = np.where(np.any(query_ids, axis=0))[0]
    marg_spn = spn.marginalize(marg_ids)

    x_range = np.linspace(spn.domains[featureId][0], spn.domains[featureId][-1], detail)
    query = np.repeat(evidence, x_range.shape[0], axis=0)
    query[:,featureId] = x_range
    y_range = np.exp(marg_spn.eval(query))

    plt.plot(x_range, y_range)
    plt.ylim(ymin=0)
    plt.ylim(ymax=np.amax(y_range) + 1)
    Scatter(
        x = random_x,
        y = random_y0,
        mode = 'lines',
        name = 'lines'
    )
    data = [Heatmap(z=result,
            y=np.linspace(spn.domains[featureId_y][0], spn.domains[featureId_y][-1], detail) if not y_cat else y_names,
            x=np.linspace(spn.domains[featureId_x][0], spn.domains[featureId_x][-1], detail) if not x_cat else x_names,
            colorbar=ColorBar(
                title='Colorbar'
            ),
            colorscale='Hot')]
    layout = dict(width=450, 
                  height=450,
                  xaxis=dict(title=spn.featureNames[featureId_y]),
                  yaxis=dict(title=spn.featureNames[featureId_x])
                 )

    if fname is None:
        return {'data': data, 'layout': layout}
    else:
        raise NotImplementedError


def plot_error_bar(names, means, stds, feature, fname=None):
    x = names
    y = means
    e = stds
    
    data = [Scatter(
        x=x,
        y=y,
        error_y=dict(
            type='data',
            array=e,
            visible=True),
        mode='markers',)]

    layout = dict(width=450, 
                  height=450,
                  xaxis=dict(title='Nodes'),
                  yaxis=dict(title=feature),
                 )

    if fname is None:
        return {'data': data, 'layout': layout}
    else:
        raise NotImplementedError
    

def matshow(matrix, title=None, x_labels=None, y_labels=None, fname=None):
    mat = np.copy(matrix)
    # idx = np.diag_indices_from(mat)
    # mat[idx] = 0

    x_range = np.arange(0, mat.shape[0], 1) + 0.5
    
    data = [Heatmap(z=mat,
            y=x_labels,
            x=y_labels,
            zmin=-1,
            zmax=1,
            colorbar=ColorBar(
                title='Colorbar'
            ),
            colorscale='RdBu')]
    layout = dict(width=450, 
                  height=450,
                  title=title,
                 )

    if fname is None:
        return {'data': data, 'layout': layout}
    else:
        raise NotImplementedError


def plot_correlated_features(spn, threshold=0.7, detail=100, fname=None):
    root = spn.root
    corr = get_correlation_matrix(spn)
    x, y = np.where(np.abs(corr) > threshold)
    for i, _ in enumerate(x):
        if x[i] > y[i]:
            plot_related_features(spn, x[i], y[i], detail, fname=fname)


def plot_decision_boundary(spn, featureX, featureY, categoricalId, featureNames, detail=100):
    marg_likelihood = spn.marginalize([featureX, featureY])
    marg_product = spn.marginalize([featureX, featureY, categoricalId])

    x_range = (spn.domains[featureX][0], spn.domains[featureX][-1])
    y_range = (spn.domains[featureY][0], spn.domains[featureY][-1])
    if spn.featureTypes[featureX] == 'categorical':
        x_detail = len(spn.domains[featureX])
    else:
        x_detail = detail
    if spn.featureTypes[featureY] == 'categorical':
        y_detail = len(spn.domains[featureY])
    else:
        y_detail = detail

    categorical_range = (spn.domains[categoricalId][0], spn.domains[categoricalId][-1])

    grid = np.mgrid[x_range[0]:x_range[1]:x_detail*1j, y_range[0]:y_range[1]:y_detail*1j]
    #if grid.shape != (2, detail, detail):
    #    grid = grid[:,:detail,:detail]
    grid = grid.reshape(2,-1).T

    query = np.array([[np.nan] * spn.numFeatures] * (x_detail * y_detail))

    query[:,featureX] = grid[:,0]
    query[:,featureY] = grid[:,1]
    for i in range(categorical_range[0], categorical_range[1] + 1):
        print(i)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        query[:,categoricalId] = i
        result = np.exp(marg_product.eval(query) - marg_likelihood.eval(query))
        result = result.reshape(x_detail, y_detail)
        cax = ax.imshow(result, vmin=0, vmax=1, origin='upper', extent=[y_range[0], y_range[1], x_range[0], x_range[1]], aspect='auto', cmap=cm.RdYlGn)
        fig.colorbar(cax)
        ax.set_xlabel(featureNames[featureY])
        ax.set_ylabel(featureNames[featureX])
        plt.show()

def plot_all_leaves(spn, detail=100, dictionary=None):
    root = spn.root
    root.validate()
    for n in spn.root.children:
        spn.root = n
        if n.leaf:
            size = spn.numFeatures
            featureId = list(n.scope)[0]

            if spn.featureTypes[featureId] == 'categorical':
                enc = dictionary['features'][featureId]['encoder'].inverse_transform
                x_range = spn.domains[featureId]
                query = np.array([[np.nan] * spn.numFeatures] * len(x_range))
                query[:,featureId] = x_range
                y_range = np.exp(n.eval(query)).reshape(len(x_range))
                plt.bar(x_range, height=y_range)
                plt.xticks(x_range, enc(x_range))
            else:
                x_range = np.linspace(spn.domains[featureId][0], spn.domains[featureId][-1], detail)
                query = np.array([[np.nan] * spn.numFeatures] * detail)
                query[:,featureId] = x_range
                y_range = np.exp(n.eval(query)).reshape(detail)

                plt.plot(x_range, y_range)
                plt.ylim(ymin=0)
                plt.ylim(ymax=np.amax(y_range)*1.1)
            plt.show()
            plt.clf()
        else:
            plot_all_leaves(spn, dictionary=dictionary, detail=detail)
        spn.root = root

def mpe_plot(spn, query_feature, target_feature, detail=100, fname=None):
    marg = spn.marginalize([query_feature, target_feature])
    x_range = (spn.domains[query_feature][0], spn.domains[query_feature][-1])
    x = np.linspace(x_range[0], x_range[1], detail)
    query = np.zeros((detail, spn.numFeatures))
    query[:,query_feature] = x
    query[:,target_feature] = np.nan

    y = marg.mpe_eval(query)[1][:,target_feature]

    plt.plot(x,y)
    plt.show()


def plot_table(header, cells):
    trace = [Table(
        header=dict(values=header,
            line = dict(color='#7D7F80'),
            fill = dict(color='#a1c3d1'),
            align = ['left'] * 5),
        cells=dict(values=cells,
            line = dict(color='#7D7F80'),
            fill = dict(color='#EDFAFF'),
            align = ['left'] * 5))
        ]
    height = min(200 + len(cells[0]) * 40, 1000)
    fig = dict(data=trace, layout=dict(height=height, width=1000))
    return fig


def plot_categorical_conditional(spn, d, featureId, categoricalId, detail=1000, fname=None):
    joint = spn.marginalize([featureId, categoricalId])
    marg = spn.marginalize([categoricalId])

    def proba(query):
        return np.exp(joint.eval(query) - marg.eval(query))

    x_range = spn.domains[featureId]
    x_is_cat = spn.featureTypes[featureId] == 'categorical'
    x_detail = detail if not x_is_cat else len(spn.domains[featureId])
    x_names = None if not x_is_cat else d['features'][featureId]['encoder'].inverse_transform(np.array(x_range))

    y_range = spn.domains[categoricalId]
    y_names = d['features'][categoricalId]['encoder'].inverse_transform(np.array(y_range))
    grid = np.mgrid[x_range[0]:x_range[-1]:x_detail*1j, y_range[0]:y_range[-1]:len(spn.domains[categoricalId])*1j]
    grid = grid.reshape(2,-1).T
    
    query = np.zeros((grid.shape[0], spn.numFeatures))
    query[:,:] = np.nan
    query[:,featureId] = grid[:,0]
    query[:,categoricalId] = grid[:,1]

    result = proba(query)
    result = np.exp(result)
    result.shape = (x_detail, len(spn.domains[categoricalId]))
    
    # plot
    data = [Heatmap(z=result,
            x=y_names,
            y=np.linspace(spn.domains[featureId][0], spn.domains[featureId][-1], detail) if not x_is_cat else x_names,
            colorbar=ColorBar(
                title='Colorbar'
            ),
            colorscale='Hot')]
    layout = dict(width=450, 
                  height=450,
                  title='COnditional probability function of "{}" and "{}"'.format(spn.featureNames[featureId], spn.featureNames[categoricalId]),
                  yaxis=dict(title=spn.featureNames[featureId], autotick=True),
                  xaxis=dict(title=spn.featureNames[categoricalId], autotick=True)
                 )

    if fname is None:
        return {'data': data, 'layout': layout}
    else:
        raise NotImplementedError


def plot_explanation_vectors(gradients, discretize):
    binsize = discretize[1][1] - discretize[1][0]
    layout = dict(width=450, 
                  height=450,)
    data = [Histogram(x=gradients,
                      autobinx=False,
                      xbins=dict(
                          start=discretize[1][0],
                          end=discretize[1][-1],
                          size=binsize
                      ))]
    return {'data': data, 'layout': layout}


def plot_cat_explanation_vector(plot_data):
    data = []
    # dirty hack which produces the centers for the bar chart
    x = (np.arange(10)-4.5)/5
    for plot in plot_data:
        binsize = plot[1][1][1] - plot[1][1][0]
        data.append(Histogram(x=plot[0],
                  autobinx=False,
                  name = plot[2],
                  xbins=dict(
                      start=plot[1][1][0],
                      end=plot[1][1][-1],
                      size=binsize,
                  )))

    layout = dict(width=450, height=450, barmode='stack')
    return {'data': data, 'layout': layout}
