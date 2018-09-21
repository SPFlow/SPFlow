import numpy as np

from spn.algorithms.stats.Expectations import get_means, get_variances
from spn.algorithms.Marginalization import marginalize
from spn.algorithms.Inference import log_likelihood
from spn.structure.Base import Sum


def cluster_anova(spn):
    if not isinstance(spn, Sum):
        raise ValueError('Only sum nodes represent clustering')

    all_means = []
    all_vars = []
    all_probs = spn.weights
    real_var = get_variances(spn).reshape(-1)

    for node in spn.root.children:
        var = get_variances(node).reshape(-1)
        mean = get_means(node).reshape(-1)
        all_vars.append(var)
        all_means.append(mean)
    all_vars = np.array(all_vars)
    all_probs = np.array(all_probs).reshape(-1, 1)
    total_var = np.sum(all_vars * all_probs, axis=0)
    result = 1 - (total_var/real_var)
    return result


def cluster_mean_var_distance(nodes, spn):
    all_means = []
    all_vars = []
    real_var = get_variances(spn).reshape(-1)
    real_mean = get_means(spn).reshape(-1)
    for node in nodes:
        var = get_variances(node).reshape(-1)
        mean = get_means(node).reshape(-1)
        all_vars.append(var)
        all_means.append(mean)
    all_vars = np.array(all_vars)
    all_means = np.array(all_means)

    return (all_vars - all_vars.mean(axis=0))/np.sqrt(all_vars.var(axis=0)), \
           (all_means - real_mean)/np.sqrt(real_var)


def categorical_nodes_description(spn, context):
    categoricals = context.get_categoricals()
    num_features = len(spn.scope)
    total_analysis = {}
    for cat in categoricals:
        marg_total = marginalize(spn, [cat])
        categorical_probabilities = []
        for i, n in enumerate(spn.children):
            node_weight = np.log(spn.weights[i])
            node_probabilities = []
            for cat_instance in context.get_domains_by_scope([cat])[0]:
                marg = marginalize(n, [cat])
                query = np.zeros((1, num_features))
                query[:, :] = np.nan
                query[:, cat] = cat_instance
                proba = np.exp(log_likelihood(marg, query) + node_weight - log_likelihood(marg_total, query)).reshape(-1)
                node_probabilities.append(proba)
            categorical_probabilities.append(np.array(node_probabilities))
        total_analysis[cat] = np.sum(np.array(categorical_probabilities), axis=2)

    node_categoricals = {}
    for cat in categoricals:
        node_categoricals[cat] = {}
        node_categoricals[cat]['contrib'] = []
        node_categoricals[cat]['explained'] = []
        for cat_instance in [int(c) for c in context.get_domains_by_scope([cat])[0]]:
            probs = total_analysis[cat]
            # TODO: That threshold needs some evidence or theoretical grounding
            contrib_nodes = np.where(probs[:, cat_instance]/(np.sum(probs, axis=1)) > 0.4)
            explained_probs = np.sum(probs[contrib_nodes], axis=0)
            node_categoricals[cat]['contrib'].append(contrib_nodes)
            node_categoricals[cat]['explained'].append(explained_probs)
    return node_categoricals, total_analysis
