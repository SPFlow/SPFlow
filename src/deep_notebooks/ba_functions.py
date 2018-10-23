import itertools
import math
import numpy as np
import pandas as pd
import scipy.integrate as integrate
from sklearn.preprocessing import LabelEncoder

from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear
from spn.structure.Base import Context

from spn.algorithms.LearningWrappers import learn_mspn
from spn.algorithms.Inference import likelihood


def query(spn, instance):
    '''
    simple query wrapper which performs exponentiation on an spn query return
    :param spn: a valid spn used for the query
    :param instance: an instantiation of all rvs
    :return: exponential of the query result
    '''
    return np.exp(spn.root.eval(instance))


def predict(spn, feature, instances):
    '''
    Returns the most probable configuration of the feature given the instance as evidence

    :param spn: the spn used for the query
    :param feature: the feature to predict (must be a categorical feature)
    :param instances: the query instance
    :return: argmax_x P(x|instance)
    '''
    return np.argmax(predict_proba(spn, feature, instances), axis=1)


def get_variance(node, numFeatures):
    '''
    returns the variance of all RVs encapsuled by the node
    :param node: the node (most often an spn.root)
    :param numFeatures: the amount of features encapsuled by the node
    :return: the variance of all RVs
    '''
    return node.moment(2, numFeatures) - node.moment(1, numFeatures) ** 2


def save_dataset(dataset, file_location):
    '''
    Wrapper to save a preformated pandas
    :param dataset:
    :param file_location:
    :return:
    '''
    values = dataset.data
    targets = dataset.target.reshape(np.size(dataset.target), 1)
    whole = np.append(values, targets, axis=1)
    np.savetxt(file_location, whole, delimiter=",")


def learn_spn(dataset="data/iris", precision=25, independence=0.1, header=0, date=None, isotonic=False, histogram=True, types=False):
    '''
    Learning wrapper for the jupyter notebooks. Performs some preliminary feature analysis on the given dataset and prepares the data
    for the mspn
    :param dataset: file location of the dataset
    :param precision: the precision used for calculating the notebooks spn TODO: find better name
    :param independence: the maximum dependence a statistical independence test
                         may show to split columns
    :param header: the location of the csv header
    :param date: location of datetime columns in the csv
    :param isotonic: whether to use isotonic nodes
    :param histogram: whether to ue histogram nodes
    :param types: whether the csv contains type annotations
    :return: a valid spn and a data dictionary containing preprocessing information
    '''
    skiprows = [1] if types else []
    df = pd.read_csv(dataset, delimiter=",", header=header, parse_dates=date, skiprows=skiprows)
    df = df.dropna(axis=0, how='any')
    feature_names = df.columns.values.tolist() if header == 0 else ["X_{}".format(i) for i in range(len(df.columns))]
    
    dtypes = df.dtypes

    def to_feature_types(types):
        feature_types = []
        for feature_type in types:
            if feature_type.kind == 'O':
                feature_types.append(Categorical)
            elif feature_type.kind == 'f':
                feature_types.append(PiecewiseLinear)
            elif feature_type.kind == np.dtype('i'):
                feature_types.append(PiecewiseLinear)
            else:
                feature_types.append(PiecewiseLinear)
        return feature_types

    if not types:
        feature_types = to_feature_types(dtypes)

    # TODO: Build Context wrapper according to README.md, this should work pretty well

    data_dictionary = {
            'features': [{"name": name,
                          "type": typ,
                          "pandas_type": dtypes[i]}
                         for i, (name, typ)
                         in enumerate(zip(feature_names, feature_types))],
            'num_entries': len(df)
            } 

    idx = df.columns
    
    for id, name in enumerate(idx):
        if feature_types[id] == Categorical:
            lb = LabelEncoder()
            data_dictionary['features'][id]["encoder"] = lb
            df[name] = df[name].astype('category')
            df[name] = lb.fit_transform(df[name])
            data_dictionary['features'][id]["values"] = lb.transform(lb.classes_)
        if dtypes[id].kind == 'M':
            df[name] = (df[name] - df[name].min())  / np.timedelta64(1,'D')

    data = np.array(df)

    # TODO: PiecewiseLinear Types do not work like the ParametricNodes

    spn = learn_mspn(data,
                     Context(parametric_types=feature_types).add_domains(data),
                     cols="rdc",
                     rows="kmeans",
                     min_instances_slice=200,
                     threshold=0.3,
                     ohe=False,
                     leaves=None)
    
    spn.name = dataset
    return spn, data_dictionary


def learn_with_cross_valid(search_grid, dataset="data/iris", header=0, date=None, isotonic=True):
    '''
    A wrapper to learn a crossvalidated spn
    :param search_grid:
    :param dataset:
    :param header:
    :param date:
    :param isotonic:
    :return:
    '''
    spn, d = learn_spn(dataset=dataset, header=header, date=date, isotonic=isotonic)
    valid = load_dataset(dataset + '.valid', d)
    precisions = np.linspace(search_grid['pre'][0], search_grid['pre'][1], search_grid['pre'][2])
    independencies = np.linspace(search_grid['ind'][0], search_grid['ind'][1], search_grid['ind'][2])
    spns = np.array([[learn_spn(dataset=dataset, precision=i, independence=j, header=header, date=date, isotonic=isotonic)[0] for i in precisions] for j in independencies])
    log_likelihood = np.array([[np.sum(s.root.eval(valid)) for s in spn] for spn in spns])
    max_idx = np.unravel_index(np.argmax(log_likelihood), log_likelihood.shape)
    print(max_idx)
    
    return spns[max_idx], d, log_likelihood[max_idx]


def spn_query_id(spn, index, value, query=None):
    '''

    :param spn:
    :param index:
    :param value:
    :param query:
    :return:
    '''
    if not query:
        query = np.array([[np.nan] * spn.numFeatures])
    query[:,index] = value
    return spn.root.eval(query)


def get_moment(spn, query_id, moment=1, evidence=None, detail=1000):
    root = spn.root
    # find the marginalization and conditioning
    marg_ids = [query_id]
    if evidence is not None:
        query_ids = np.isfinite(evidence)
        query_ids[:,query_id] = True
        marg_ids = np.where(np.any(query_ids, axis=0))[0]
    marg_spn = spn.marginalize(marg_ids)
    spn.root = marg_spn

    mean = integrate.quad(lambda x: np.exp(spn_query_id(spn, query_id, x)) * x, spn.domains[query_id][0], spn.domains[query_id][1])[0]
    spn.root = root
    if moment == 1:
        return mean
    return integrate.quad(lambda x: np.exp(spn_query_id(spn, query_id, x)) * (x - mean) ** moment, spn.domains[query_id][0], spn.domains[query_id][1])[0]

def validate_feature(spn, featureId, precision=0.1):
    if spn.families[featureId] == ('categorical'):
        pass
    else:
        lw = spn.domains[featureId][0]
        up = spn.domains[featureId][-1]

        x_range = np.arange(lw, up-precision, precision)
        z = np.array((1))

        for x in x_range:
            z = np.append(z, precision * np.exp(func_from_spn(spn, featureId)(x)))

        return np.sum(z)


def get_feature_entropy(spn, featureId, precision=0.1, numeric=False):
    lw = spn.domains[featureId][0]
    up = spn.domains[featureId][-1]
    if numeric:
        x_range = np.arange(lw_0, up_0-precision, precision)
        H_x = np.zeros((1,1))
        for x in x_range:
            H_x = np.sum(H_x, func_from_spn(spn, featureId)(x))
        return -1 * H_x
    else:
        return integrate.quad(lambda x: -1 * (func_from_spn(spn, featureId)(x) * np.exp(func_from_spn(spn, featureId)(x))), lw, up)


def get_mutual_information(spn, featureIdx, precision=0.1, numeric=False):
    lw_0 = spn.domains[featureIdx[0]][0]
    lw_1 = spn.domains[featureIdx[1]][0]
    up_0 = spn.domains[featureIdx[0]][-1]
    up_1 = spn.domains[featureIdx[1]][-1]
    
    def joined(spn, featureIdx):
        size = spn.numFeatures
        marg_spn = spn.marginalize(featureIdx)
        query = np.zeros((1, size))

        def func(x, y):
            query[:, featureIdx[0]] = x
            query[:, featureIdx[1]] = y
            return marg_spn.eval(query)

        return func

    p_x = func_from_spn(spn, featureIdx[0])
    p_y = func_from_spn(spn, featureIdx[1])
    p_x_y = joined(spn, featureIdx)

    mi = lambda x, y: np.exp(p_x_y(x, y)) * np.log2(np.exp(1)) * (p_x_y(x, y) - p_x(x) - p_y(y))

    if numeric:
        x_range = np.arange(lw_0, up_0-precision, precision)
        y_range = np.arange(lw_1, up_1-precision, precision)

        z = np.array([])

        for x in x_range:
            for y in y_range:
                z = np.append(z, mi(x, y)*(precision*precision))

        return np.sum(z)
    else:
        return integrate.dblquad(lambda y, x: mi(x,y), lw_0, up_0, lambda x: lw_1, lambda x: up_1)


def get_feature_decomposition(spn, feature, instance):
    """
    spn: a valid tfspn.SPN
    feature: the feature for which the influence should be computed
    instance: a list of query instances over which the difference is to be computed 
                for each feature. The final reported weight for each feature is the 
                mean over all query instances

    computes the log information difference, based on Robnik et.al.

    log(p(y|A)) - log(p(y|A/a))
    """
    marginalized_likelihood = [i for i in range(spn.numFeatures) if i != feature]
    marg_likelihood = spn.marginalize(marginalized_likelihood)
    influence = np.zeros((instance.shape[0], spn.numFeatures))
    for i in range(spn.numFeatures):
        if i != feature:
            marginalize_decomposition = [j for j in range(spn.numFeatures) if j != i]
            marginalize_all = [j for j in range(spn.numFeatures) 
                    if j != i and j != feature]
            marg_decomposition = spn.marginalize(marginalize_decomposition)
            marg_all = spn.marginalize(marginalize_all)
            influence[:,i] = spn.root.eval(instance) - marg_likelihood.eval(instance) - marg_decomposition.eval(instance) + marg_all.eval(instance)
    return influence


def get_gradient(spn, instance, fixed_instances=[], step_size=0.0001):
    """
    implements a quick and dirty approach to calculating gradients of the spn
    """
    instances = np.repeat(instance, spn.numFeatures, axis=0)
    h = np.identity(spn.numFeatures) * step_size
    h[fixed_instances] = np.zeros(spn.numFeatures)
    query_minus = instances - h
    query_plus = instances + h
    prob_minus = np.exp(spn.root.eval(query_minus))
    prob_plus = np.exp(spn.root.eval(query_plus))
    return (prob_plus-prob_minus)/step_size*2


def get_strongly_related_features(spn):
    """
    spn: a valid tfspn.SPN object

    Tries to see how strongly two features are connected by analyzing the level
    of the spn that generally seperates them. The interconnection is measured
    between [0,1], 0 meaning that they are seperated as soon as possible and 1 
    that the two features are never separated until the final split.
    """
    
    root = spn.root
    root.validate()
    features = root.scope

    combinations = list(itertools.combinations(features, 2))

    def recurse(node, feature1, feature2, depth, depth_list):
        features_in_children = any(((feature1 not in c.scope) != (feature2 not in c.scope) for c in node.children))
        if features_in_children:
            depth_list.append(depth)
        else:
            for c in node.children:
                recurse(c, feature1, feature2, depth+1, depth_list)
        return depth_list

    return {c: np.array(recurse(root, c[0], c[1], 0, [])).mean() for c in combinations}


def get_examples_from_nodes(spn):
    query = np.array([[np.nan] * spn.numFeatures])
    return [c.mpe_eval(query) for c in spn.root.children]


def get_category_examples(spn, categories):
    samples = {}
    for categoryId in categories:
        name = categoryId
        samples[name] = []
        for example in categories[categoryId][1]:
            query = np.array([[np.nan] * spn.numFeatures])
            query[:,categoryId] = example
            categorical_name = categories[categoryId][0].inverse_transform(example)
            row = [categorical_name] + np.round(spn.root.mpe_eval(query)[1], 2).tolist()[0]
            samples[name].append(row)
    return samples


def get_covariance_matrix(spn):
    size = spn.numFeatures
    joined_means = spn.root.joined_mean(size)
    means = joined_means.diagonal().reshape(1, size)
    squared_means = means.T.dot(means)
    covariance = joined_means - squared_means

    diagonal_correlations = spn.root.moment(2, size) - spn.root.moment(1, size) ** 2
    idx = np.diag_indices_from(covariance)
    covariance[idx] = diagonal_correlations

    return covariance


def get_correlation_matrix(spn):
    size = spn.numFeatures
    covariance = get_covariance_matrix(spn)
    sigmas = np.sqrt(spn.root.moment(2, size) - spn.root.moment(1, size) ** 2).reshape(1,size)
    sigma_matrix = sigmas.T.dot(sigmas)
    correlations = covariance / sigma_matrix

    return correlations

def get_node_feature_probability(spn, featureIdx, instance):
    root = spn.root
    probabilities = np.array([])
    for c in spn.root.children:
        # print("x")
        spn.root = c
        # print(func_from_spn(spn, featureIdx)(instance))
        probabilities = np.append(probabilities, np.exp(func_from_spn(spn, featureIdx)(instance)))
        spn.root = root
    return probabilities

def calculate_overlap(inv, invs, area=0.9):
    overlap = []
    for i in invs:
        #print(overlap)
        covered = 0
        smaller = inv if inv[0] < i[0] else i
        bigger = inv if smaller == i else i
        if smaller[1] > bigger[0]:
            covered += smaller[1] - bigger[0]
            if smaller[1] > bigger[1]:
                covered -= smaller[1] - bigger[1]
        if covered/(i[1]-i[0]) >= area and covered/(inv[1]-inv[0]) >= area:
            overlap.append(i)
    return overlap


def predict_proba(spn, feature, query):
    from concurrent.futures import ThreadPoolExecutor
    domain = np.linspace(spn.domains[feature][0], spn.domains[feature][-1], len(spn.domains[feature]))
    proba = []
    marg = spn.marginalize([i for i in range(spn.numFeatures) if i is not feature])
    def predict(q):
        _query = np.copy(q).reshape((1, spn.numFeatures))
        _query = np.repeat(_query, len(spn.domains[feature]), axis=0)
        _query[:, feature] = domain
        prediction = np.exp(spn.root.eval(_query))
        prediction /= np.sum(prediction)
        return prediction

    with ThreadPoolExecutor(max_workers = 50) as thread_executor:
        results = []
        for q in query:
            #all_pos = list(enumerate(pos))
            #results = thread_executor.map(calculate_feature, all_pos)
            results.append(thread_executor.submit(predict, q))
        proba.append([r.result() for r in results])
    return np.array(proba)[0]


def spn_predict_proba_single_class(spn, feature, instance, query):
    proba = spn_predict_proba(spn, feature, query)
    y_true = proba[:,instance]
    return y_true


def load_dataset(data, dictionary, types=False, header=0, date=False):
    skiprows = [1] if types else []
    df = pd.read_csv(data, delimiter=",", header=header, parse_dates=date, skiprows=skiprows)
    categoricals = (i for i, d in enumerate(dictionary['features']) 
                if d['type'] == 'categorical')
    for i in categoricals:
        df.iloc[:,i] = dictionary['features'][i]['encoder'].transform(df.iloc[:,i])
    return df.as_matrix()


def get_mean(spn):
    return spn.root.moment(1, spn.numFeatures)


def get_var(spn):
    return spn.root.moment(2, spn.numFeatures) - spn.root.moment(1, spn.numFeatures) ** 2


def shapley_variance_of_nodes(spn, sample_size):
    from datetime import datetime
    root = spn.root
    startTime = datetime.now()
    total_var = get_var(spn)
    num_nodes = len(spn.root.children)
    len_permutations = math.factorial(num_nodes)
    node_permutations = np.array(list(itertools.permutations([i for i in range(num_nodes)])))
    if sample_size < len_permutations:
        sample = np.random.choice(len_permutations, sample_size)
        node_permutations = node_permutations[sample]
    node_permutations = np.array(node_permutations)
    nodes = spn.root.children
    weights = spn.root.weights
    shapley_values = np.zeros((num_nodes, spn.numFeatures))
    for i in range(num_nodes):
        pos = np.where(node_permutations==i)[1]
        query_sets = [node_permutations[n,:j] for n, j in enumerate(pos)]
        for query_set in query_sets:
            spn.root.children = np.array(nodes)[query_set]
            spn.root.weights = np.array(weights)[query_set]
            var_without = get_var(spn)
            query_set = np.append(query_set, i)
            spn.root.children = np.array(nodes)[query_set]
            spn.root.weights = np.array(weights)[query_set]
            var_with = get_var(spn)
            shapley_values[i] += (var_with - var_without)/min(sample_size, len_permutations)
    spn.root.children = nodes
    spn.root.weights = weights
    return shapley_values, datetime.now() - startTime


def feature_contribution(spn, query, categorical, sample_size = 10000):
    from concurrent.futures import ThreadPoolExecutor
    len_permutations = math.factorial(spn.numFeatures-1)
    sets = np.array(list(itertools.permutations([i for i in range(spn.numFeatures) if i != categorical])))
    if sample_size < len_permutations:
        sample = np.random.randint(len_permutations, size=sample_size)
        sets = sets[sample]
    #sets = np.array(sets)
    sample = len(sets)
    weights = spn.root.weights
    shapley_values = np.zeros((query.shape[0], spn.numFeatures))
    def calculate_feature(i, n, j, cat):
        s = sets[n,:j]
        marg_ci = spn.marginalize(np.append(s, categorical)).eval(query)
        marg_c = spn.marginalize(np.append(s[:j-1], categorical)).eval(query)
        marg_i = spn.marginalize(s[:j]).eval(query)
        if len(s[:j-1]) == 0:
            marg = 0
        else:
            marg = spn.marginalize(s[:j-1]).eval(query)
        return (np.exp(marg_ci - marg_i) - np.exp(marg_c - marg))
    for i in range(spn.numFeatures):
        pos = np.where(sets==i)[1] + 1
        results = []
        with ThreadPoolExecutor(max_workers = 50) as thread_executor:
            for n,j in enumerate(pos):
                results.append(thread_executor.submit(calculate_feature, i, n, j, categorical))
        results = [r.result() for r in results]
        shapley_values[:,i] = np.sum(np.array(results))/len(pos)
    return shapley_values


def node_likelihood_contribution(spn, query):
    children = spn.root.children
    children_weights = spn.root.weights
    children_log_weights = spn.root.log_weights

    nodes = spn.root.children.copy()
    weights = spn.root.weights.copy()
    log_weights = spn.root.log_weights.copy()
    weighted_nodes = list(zip(nodes, weights, log_weights))
    weighted_nodes.sort(key=lambda x: x[1])
    weighted_nodes.reverse()
    nodes = [x[0] for x in weighted_nodes]
    weights = [x[1] for x in weighted_nodes]
    log_weights = [x[2] for x in weighted_nodes]

    log_likelihood = []

    for i, n in enumerate(nodes):
        spn.root.children = nodes[:i+1]
        spn.root.weights = weights[:i+1]
        spn.root.log_weights = log_weights[:i+1]
        log_likelihood.append(np.sum(spn.root.eval(query)))

    spn.root.children = children
    spn.root.weights = children_weights
    return log_likelihood


def get_marginal_arrays(spn):
    vals = []
    tps = []
    for feature in range(spn.numFeatures):
        if spn.featureTypes[feature] != 'categorical':
            marg = spn.marginalize([feature])
            _min = spn.domains[feature][0]
            _max = spn.domains[feature][-1]
            turning_points = np.concatenate([n.x_range for n in marg.children])
            turning_points = np.unique(np.sort(np.array([p for p in turning_points if (p > _min) and (p < _max)])))
            
            query = np.zeros((len(turning_points), spn.numFeatures))
            query[:,:] = np.nan
            query[:,feature] = turning_points
            
            values = np.exp(marg.eval(query))
            tps.append(turning_points)
            vals.append(values)
    return np.array(tps), np.array(vals)

class ProbabilityTree():
    def __init__(self, value):
        self.value = value
        self.children = []

    def append(self, child):
        self.children.append(child)

def probability_tree(spn, query, categoricalId):
    categorical_values = spn.domains[categoricalId]
    _query = np.repeat(query, len(categorical_values), axis=0)
    _query[:, categoricalId] = categorical_values

def recursive_feature_contribution(spn, query, categorical):
    root = spn.root
    marg = spn.marginalize([i for i in range(spn.numFeatures) if i != categorical])
    proba = spn.root.eval(query) - marg.eval(query)
    norm = marg.eval(query)

    results = []
    node_probas = []

    for i, node in enumerate(spn.root.children):
        spn.root = node
        results.append(np.exp(root.log_weights[i] + spn.root.eval(query) - norm))
        proba_query = np.zeros((len(spn.domains[categorical]), spn.numFeatures))
        proba_query[:,categorical] = spn.domains[categorical]
        node_probas.append(np.exp(spn.marginalize([categorical]).eval(proba_query)))
        spn.root = root

    return np.array(results), np.array(node_probas)


def get_explanation_vector(spn, data, categorical):
    marg = spn.marginalize([i for i in range(spn.numFeatures) if i != categorical])
    results = []
    for d in data:
        d.shape = (1,-1)
        gradients_xy = get_gradient(spn, d, categorical)
        root = spn.root
        spn.root = marg
        gradient_x = get_gradient(spn, d, categorical)
        spn.root = root
        result_xy = np.exp(spn.root.eval(d)).reshape((-1,1))
        result_x = np.exp(marg.eval(d)).reshape((-1,1))
        results.append((gradients_xy * result_x - result_xy * gradient_x)/(result_x ** 2))
    return np.array(results).reshape(len(data),spn.numFeatures)


def gradient(spn, data, categorical):
    marg = spn.marginalize([i for i in range(spn.numFeatures) if i != categorical])
    gradients_xy = spn.root.gradient(data)
    gradient_x = marg.gradient(data)
    result_xy = np.exp(spn.root.eval(data)).reshape((-1,1))
    result_x = np.exp(marg.eval(data)).reshape((-1,1))
    return np.delete((gradients_xy * result_x - result_xy * gradient_x)/(result_x ** 2), categorical, axis = 1)


def prediction_nodes(spn, query, categorical):
    root = spn.root
    norm = spn.marginalize(
            [i for i in range(spn.numFeatures) if i != categorical]).eval(query)
    result = []
    for i, n in enumerate(spn.root.children):
        spn.root = n
        prob = spn.marginalize(
                [i for i in range(spn.numFeatures) if i != categorical]).eval(query)
        result.append(np.exp(root.log_weights[i] + prob - norm))
        spn.root = root
    return np.array(result)


def get_categorical_correlation(spn):
    categoricals = [i for i, t in enumerate(spn.featureTypes) if t == 'categorical']
    var = get_variance(spn.root, spn.numFeatures)
    var = spn.root.moment(2, spn.numFeatures) - spn.root.moment(1, spn.numFeatures) ** 2
    all_vars = []
    for cat in categoricals:
        all_probs = []
        cat_vars = []
        query = [np.nan] * spn.numFeatures
        domain = spn.domains[cat]
        for value in domain:
            query[cat] = value
            cond_spn, prob = spn.root.conditional(query)
            cond_var = get_variance(cond_spn, spn.numFeatures)
            cat_vars.append(cond_var)
            all_probs.append(np.exp(prob))
        cat_vars = np.array(cat_vars)
        all_probs = np.array(all_probs).reshape(-1, 1)
        total_var = np.sum(cat_vars * all_probs, axis = 0)
        #print(total_var)
        #print(var)
        result = 1 - (total_var/var)
        all_vars.append(result)
    all_vars = np.array(all_vars)
    assert np.all(np.logical_or(all_vars > -0.0001, np.isnan(all_vars)))
    all_vars[all_vars < 0] = 0
    return np.sqrt(all_vars)


def get_mutual_information_correlation(spn):
    categoricals = [i for i, t in enumerate(spn.featureTypes) if t == 'categorical']

    correlation_matrix = []
    
    for x in categoricals:
        x_correlation = []
        x_range = spn.domains[x]
        spn_x = spn.marginalize([x])
        query_x = np.array([[np.nan] * spn.numFeatures] * len(x_range))
        query_x[:,x] = x_range
        for y in categoricals:
            if x == y:
                x_correlation.append(1)
                continue
            spn_y = spn.marginalize([y])
            spn_xy = spn.marginalize([x,y])
            y_range = spn.domains[y]
            query_y = np.array([[np.nan] * spn.numFeatures] * len(y_range))
            query_y[:,y] = y_range
            query_xy = np.array([[np.nan] * spn.numFeatures] * (len(x_range + 1) * (len(y_range + 1))))
            xy = np.mgrid[x_range[0]:x_range[-1]:len(x_range)*1j, y_range[0]:y_range[-1]:len(y_range)*1j]
            xy = xy.reshape(2, -1)
            query_xy[:,x] = xy[0,:]
            query_xy[:,y] = xy[1,:]
            results_xy = np.exp(spn_xy.eval(query_xy))
            results_xy = results_xy.reshape(len(x_range), len(y_range))
            results_x = np.exp(spn_x.eval(query_x))
            results_y = np.exp(spn_y.eval(query_y))
            
            xx, yy = np.mgrid[0:len(x_range)-1:len(x_range)*1j, 0:len(y_range)-1:len(y_range)*1j]
            xx = xx.astype(int)
            yy = yy.astype(int)

            grid_results_x = results_x[xx]
            grid_results_y = results_y[yy]
            grid_results_xy = results_xy

            log = np.log(grid_results_xy/(np.multiply(grid_results_x,grid_results_y)))
            prod = np.prod(np.array([log, grid_results_xy]), axis = 0)

            log_x = np.log(results_x)
            log_y = np.log(results_y)

            entropy_x = -1 * np.sum(np.multiply(log_x, results_x))
            entropy_y = -1 * np.sum(np.multiply(log_y, results_y))
            
            x_correlation.append(np.sum(prod)/np.sqrt(entropy_x * entropy_y))
        correlation_matrix.append(x_correlation)

    return np.array(correlation_matrix)


def get_full_correlation(spn):
    categoricals = get_categoricals(spn)
    full_corr = get_correlation_matrix(spn)
    cat_corr = get_categorical_correlation(spn)
    cat_cat_corr = get_mutual_information_correlation(spn)
    for i, cat in enumerate(get_categoricals(spn)):
        cat_corr[:,cat] = cat_cat_corr[:,i]
    if cat_corr.size > 0:
        full_corr[categoricals,:] = cat_corr
        full_corr[:,categoricals] = cat_corr.T
    return full_corr


def get_categorical_data(spn, df, dictionary, header=1, types=False, date=False):
    categoricals = get_categoricals(spn)
    df_numerical = df.copy(deep=True)
    for i in categoricals:
        transformed = dictionary['features'][i]['encoder'].transform(df_numerical.as_matrix()[:,i])
        df_numerical.iloc[:,i] = transformed
    
    numerical_data = df_numerical.as_matrix()
    
    categorical_data = {}
    for i in categoricals:
        data = df_numerical.groupby(spn.featureNames[i])
        data = [data.get_group(x).as_matrix() for x in data.groups]
        categorical_data[i] = data

    return numerical_data, categorical_data


def get_sorted_nodes(spn):
    root = spn
    weights_nodes = [(node, weight) for node, weight in zip(root.children, root.weights)]
    weights_nodes.sort(key=lambda x: x[1])
    weights_nodes.reverse()
    nodes = [node[0] for node in weights_nodes]
    return nodes

