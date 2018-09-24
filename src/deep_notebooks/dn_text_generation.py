import random
import itertools
import os
from bisect import bisect_right as threshold
from collections import namedtuple

import numpy as np
import pandas as pd
from plotly.offline import iplot
from IPython.display import display, Image

from spn.structure.StatisticalTypes import Type
from spn.structure.Base import get_size, Leaf, get_spn_depth
from spn.algorithms.stats.Expectations import get_means, get_variances
from spn.algorithms.stats.Correlations import get_full_correlation
from spn.algorithms.stats.ClusterAnalysis import cluster_anova, cluster_mean_var_distance, categorical_nodes_description
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Gradient import conditional_gradient
from spn.algorithms.MPE import mpe, predict_mpe
from spn.algorithms.TransformStructure import Copy, assign_ids

import deep_notebooks.ba_functions as f
import deep_notebooks.dn_plot as p
from deep_notebooks.text_util import printmd, strip_dataset_name, get_nlg_phrase, deep_join, colored_string
import deep_notebooks.explanation_vector_grammar as expl_vec_grammar

# GLOBAL SETTINGS FOR THE MODULE
correlation_threshold = 0.3
explanation_vector_threshold = 0.3
num_of_feature_marginals = 10
num_of_explanation_vectors = 10
num_of_correlations = 20
feature_combinations = 2
show_conditional = True
nodes = 'all'
show_node_graphs = True
features_shown = 'all'
features_shown = 2
mean_threshold = 1
variance_threshold = 2
separation_threshold = 0.3
use_shapley = False
shapley_sample_size = 1
misclassified_explanations = 1
explanation_vector_threshold = 0
explanation_vector_classes = None
explanation_vectors_show = 'all'


Modifier = namedtuple('Modifier', ['strength', 'strength_adv', 'direction', 'neg_pos'])

CORRELATION_NLG = ['deep_notebooks/grammar', 'correlation_description.nlg']


def correlation_statement(corr, feature1, feature2):
    strength = ['weak', 'moderate', 'strong', 'very strong', 'perfect']
    strength_values = [0.3, 0.6, 0.8, 0.99]
    direction = ['decrease', 'increase']
    neg_pos = ['negative', 'positive']

    description = dict(
            strength=strength[threshold(strength_values, np.abs(corr))],
            strength_adv=strength[threshold(strength_values, np.abs(corr))]+'ly',
            direction=direction[0] if corr < 0 else direction[1],
            neg_pos=neg_pos[0] if corr < 0 else neg_pos[1],
            fx=feature1,
            fy=feature2,
            )
    sentences = [
            '"{fx}" and "{fy}" influence each other {strength_adv}. As one increases, the other {direction}s.',
            'There is a {strength} {neg_pos} dependency between "{fx}" and "{fy}".',
            'There is a {strength} linear relation between "{fx}" and "{fy}".',
            'The model shows a {strength} linear relation between "{fx}" and "{fy}".',
            'The features "{fx}" an "{fy}" have a {strength} dependency between them.'
            ]
    return sentences[random.randrange(len(sentences))].format(**description) + ' ' if np.abs(corr) > 0.25 else ''


def get_correlation_modifier(corr):
    strength = ['weak', 'moderate', 'strong', 'very strong', 'perfect']
    strength_values = [0.3, 0.6, 0.8, 0.99]
    direction = ['decrease', 'increase']
    neg_pos = ['negative', 'positive']
    strength = strength[threshold(strength_values, np.abs(corr))]
    strength_adv = strength+'ly'
    direction = direction[0] if corr < 0 else direction[1]
    neg_pos = neg_pos[0] if corr < 0 else neg_pos[1]
    return Modifier(strength, strength_adv, direction, neg_pos)


# ---------------------------------------- #
# ------ ACTUAL DESCRIPTIVE CLASSES ------ #
# ---------------------------------------- #


def introduction(spn):
    printmd('# Exploring the {} dataset'.format(strip_dataset_name(spn.name)))
    printmd('''<figure align="right" style="padding: 1em; float:right; width: 300px">
	<img alt="the logo of TU Darmstadt"
		src="deep_notebooks/images/tu_logo.gif">
	<figcaption><i>Report framework created @ TU Darmstadt</i></figcaption>
        </figure>
This report describes the dataset {} and contains general statistical
information and an analysis on the influence different features and subgroups
of the data have on each other. The first part of the report contains general
statistical information about the dataset and an analysis of the variables
and probability distributions.<br/>
The second part focusses on a subgroup analysis of the data. Different
clusters identified by the network are analyzed and compared to give an
insight into the structure of the data. Finally the influence different
variables have on the predictive capabilities of the model are analyzes.<br/>
The whole report is generated by fitting a sum product network to the
data and extracting all information from this model.'''.format(strip_dataset_name(spn.name)))


def data_description(context, df):
    '''
    Returns a general overview of the data

    :param context: Context: the context object for the data
    :param df: np.array: the data
    :return: a textual data description
    '''
    num_features = len(context.parametric_types)
    feature_names = context.feature_names
    feature_types = context.parametric_types
    printmd('')
    printmd('The dataset contains {} entries'.format(len(df)) +
            ' and is comprised of {} features, which are "{}".'.format(
                num_features, '", "'.join(feature_names)))
    categories = {'cont': [x for
                      i, x in enumerate(feature_names) if
                           feature_types[i].type == Type.REAL],
                  'cat':  [x for
                      i, x in enumerate(feature_names) if
                           feature_types[i].type == Type.CATEGORICAL],
                  'dis':  [x for
                      i, x in enumerate(feature_names) if
                           feature_types[i].type == Type.COUNT],}
    desc = ''
    if len(categories['cont']) == 1:
        desc += '"{cont}" is a continuous feature. '
    if len(categories['cont']) > 1:
        desc += '"{cont}" are continuous features, while '
    if len(categories['dis']) == 1:
        desc += '"{dis}" is a discrete feature. '
    if len(categories['dis']) > 1:
        desc += '"{dis}" are discrete features. '
    if len(categories['cont']) and len(categories['dis']):
        if len(categories['cat']) == 1:
            desc += 'Finally "{cat}" is a categorical features. '
        if len(categories['cat']) > 1:
            desc += 'Finally "{cat}" are categorical feature. '
    else:
        if len(categories['cat']) == 1:
            desc += '"{cat}" are categorical features. '
        if len(categories['cat']) > 1:
            desc += '"{cat}" is a  categorical feature. '
    printmd(desc.format(
                cont='", "'.join(categories['cont']),
                cat='", "'.join(categories['cat']),
                dis='", "'.join(categories['dis'])) + \
            'Continuous and discrete features were approximated with piecewise\
            linear density functions, while categorical features are represented by \
            histogramms of their probability.')


def means_table(spn, context):
    num_features = len(context.parametric_types)

    feature_scope = set((i for i, _ in enumerate(context.feature_names)))
    evidence_scope = set()
    evidence = None

    means = get_means(spn)
    vars = get_variances(spn)
    stds = np.sqrt(vars)
    header = ['Feature', 'Mean', 'Variance', 'STD']
    cells = [context.feature_names,
             [np.round(means[:, i], 2) for i in range(num_features)],
             [np.round(vars[:, i], 2) for i in range(num_features)],
             [np.round(stds[:, i], 2) for i in range(num_features)]]
    plot = p.plot_table(header, cells)
    iplot(plot)


def show_feature_marginals(spn, dictionary):
    all_features = list(spn.full_scope)
    if features_shown == 'all':
        shown_features = all_features
    elif isinstance(features_shown, int):
        num_choices = min(features_shown, len(all_features))
        shown_features = random.sample(all_features, k=num_choices)
    else:
        shown_features = features_shown

    for i in shown_features:
        x = p.plot_marginal(spn, i, dictionary=dictionary)
        iplot(x)


def correlation_description(spn, dictionary):
    context = dictionary['context']
    features = context.feature_names
    high_correlation = correlation_threshold
    categoricals = context.get_categoricals()
    non_categoricals = [i for i in spn.full_scope if i not in categoricals]
    corr = get_full_correlation(spn, context)
    labels = features
    iplot(p.matshow(corr, x_labels=labels, y_labels=labels))

    idx = np.where(np.abs(corr) > high_correlation)

    phrases = []
    for i in range(corr.shape[0]):
        correlated_features = [j for j in range(corr.shape[1]) if i > j and np.abs(corr[i,j]) > high_correlation]
        modifiers = [get_correlation_modifier(corr[i, j]) for j in correlated_features]
        counter = 0
        while counter < len(modifiers):
            x = labels[i]
            y = labels[correlated_features[counter]]
            phrase = get_nlg_phrase(*CORRELATION_NLG)
            if '{z}' in phrase:
                if counter == len(modifiers) - 1:
                    continue
                z = labels[counter + 1]
                mod1 = modifiers[counter]
                mod2 = modifiers[counter + 1]
                if ('but' in phrase or 'while' in phrase) and mod1.strength == mod2.strength:
                    phrase = phrase.replace(', but', ', and')
                    phrase = phrase.replace(', while', ', and')
                if 'On the other hand' in phrase and mod1.strength == mod2.strength:
                    continue
                phrase = phrase.format(
                        x=x,
                        y=y,
                        z=z,
                        strength=mod1.strength,
                        strength_adv=mod1.strength_adv,
                        strength_2=mod2.strength,
                        strength_2_adv=mod2.strength_adv,
                        direction=mod1.direction,
                        neg_pos_1=mod1.neg_pos,
                        neg_pos_2=mod2.neg_pos)
                counter += 2
            else:
                mod1 = modifiers[counter]
                phrase = phrase.format(
                        x=x,
                        y=y,
                        strength=mod1.strength,
                        strength_adv=mod1.strength_adv,
                        direction=mod1.direction,
                        neg_pos=mod1.neg_pos)
                counter += 1
            phrases.append(phrase)
    if not phrases:
        printmd('No features show more then a very weak correlation.')
    else:
        printmd(deep_join(phrases, ' ') + '\n\nAll other features do not have more then a very weak correlation.')
    return corr


def categorical_correlations(spn, dictionary):
    context = dictionary['context']
    categoricals = context.get_categoricals()
    corr = get_full_correlation(spn, context)
    num_features = len(spn.scope)
    feature_names = context.feature_names

    all_combinations = [(i, j) for i, j in
                        itertools.product(range(num_features),
                                          range(num_features)) if
                        i > j and np.abs(corr[i, j]) > correlation_threshold]
    if isinstance(feature_combinations, int):
        num_choices = min(feature_combinations, len(all_combinations))
        shown_combinations = random.sample(all_combinations, k=num_choices)
    elif feature_combinations == 'all':
        shown_combinations = all_combinations
    else:
        shown_combinations = feature_combinations

    for cat_counter, cat in enumerate(
            set([combination[0] for combination in shown_combinations])):
        for i in [combination[1] for combination in shown_combinations if
                  combination[0] == cat]:
            phrase = get_nlg_phrase(*CORRELATION_NLG)
            while '{z}' in phrase or 'As' in phrase or 'linear' in phrase:
                phrase = get_nlg_phrase(*CORRELATION_NLG)
            strength = ['weak', 'moderate', 'strong', 'very strong', 'perfect']
            strength_values = [0.3, 0.6, 0.8, 0.99]
            strength_descr = strength[
                threshold(strength_values, np.abs(corr[cat, i]))]
            strength_adv = strength_descr + 'ly'
            if show_conditional:
                iplot(p.plot_related_features(spn, i, cat,
                                              dictionary=dictionary))
            printmd(phrase.format(
                x=feature_names[cat],
                y=feature_names[i],
                strength=strength_descr,
                strength_adv=strength_adv,
                direction='',
                neg_pos=''))


def node_introduction(spn, nodes, context):
    root = spn
    desc = get_node_description(spn, spn, len(spn.scope))
    printmd('The SPN contains {} clusters.\n'.format(desc['num']))
    printmd('These are:')
    for i, d in enumerate(desc['nodes']):
        node_description = '- {}, representing {}% of the data.\n'.format(
            d['short_descriptor'], np.round(d['weight'] * 100, 2))
        if d['quick'] != 'shallow':
            if show_node_graphs:
                graph, visual_style = p.plot_graph(spn=nodes[i], fname="deep_notebooks/images/node" + str(
                    i) + ".png", context=context)
            node_description += '  - The node has {} children and {} descendants,\
                    resulting in a remaining depth of {}.\n'.format(
                d['num_children'], d['size'], d['depth'])
            printmd(node_description)
            if show_node_graphs:
                display(Image(filename="deep_notebooks/images/node" + str(i) + ".png", retina=True))
        else:
            break
    remaining = 0
    while i < len(desc['nodes']):
        d = desc['nodes'][i]
        remaining += d['weight']
        i += 1
    node_desc = 'The remaining {}% of the data is captured by {} shallow node'.format(
        np.round(remaining * 100, 2), desc['shallow'])
    if desc['shallow'] > 1:
        node_desc += 's.'
    else:
        node_desc += '.'
    if desc['shallow'] > 0:
        printmd(node_desc)
    printmd('The node representatives are the most likely data points for each node.\
            They are archetypal for what the node represents and what subgroup of\
            the data it encapsulates.')

    header = context.feature_names
    representatives = np.array(
        [np.round(d['representative'][0], 2) for d in desc['nodes']])
    cells = representatives.T
    plot = p.plot_table(header, cells)
    # plot['layout']['title'] = 'The representatives (most likely instances) of each node'
    iplot(plot)
    spn.root = root


def get_node_description(spn, parent_node, size):
    root = spn
    # parent_node.validate()
    parent_type = type(parent_node).__name__
    node_descriptions = dict()
    node_descriptions['num'] = len(parent_node.children)
    nodes = list()
    for i, node in enumerate(parent_node.children):
        node_spn = Copy(node)
        assign_ids(node_spn)
        node_dir = dict()
        node_dir['weight'] = parent_node.weights[i] if parent_type == 'Sum' else 1
        node_dir['size'] = get_size(node) - 1
        node_dir['num_children'] = len(node.children) if not isinstance(node, Leaf) else 0
        node_dir['leaf'] = isinstance(node, Leaf)
        node_dir['type'] = type(node).__name__ + ' Node'
        node_dir['split_features'] = [list(c.scope) for c in node.children] if not isinstance(node, Leaf) else node.scope
        node_dir['split_features'].sort(key=lambda x: len(x))
        node_dir['depth'] = get_spn_depth(node)
        node_dir['child_depths'] = [get_spn_depth(c) for c in node.children]

        descriptor = node_dir['type']
        if all((d == 0 for d in node_dir['child_depths'])):
            descriptor = 'shallow ' + descriptor
            node_dir['quick'] = 'shallow'
        elif len([d for d in node_dir['child_depths'] if d == 0]) == 1:
            node_dir['quick'] = 'split_one'
            descriptor += ', which separates one feature'
        else:
            node_dir['quick'] = 'deep'
            descriptor = 'deep ' + descriptor
        descriptor = 'a ' + descriptor
        node_dir['descriptor'] = descriptor
        node_dir['short_descriptor'] = descriptor
        node_dir['representative'] = mpe(node_spn, np.array([[np.nan] * size]))
        nodes.append(node_dir)
    node_descriptions['shallow'] = len([d for d in nodes if d['quick'] == 'shallow'])
    node_descriptions['split_one'] = len([d for d in nodes if d['quick'] == 'split_one'])
    node_descriptions['deep'] = len([d for d in nodes if d['quick'] == 'deep'])
    nodes.sort(key=lambda x: x['weight'])
    nodes.reverse()
    node_descriptions['nodes'] = nodes
    spn.root = root
    return node_descriptions


def show_node_separation(spn, nodes, context):
    categoricals = context.get_categoricals()
    all_features = spn.scope
    feature_names = context.feature_names

    if features_shown == 'all':
        shown_features = all_features
    elif isinstance(features_shown, int):
        num_choices = min(features_shown, len(all_features))
        shown_features = random.sample(all_features, k=num_choices)
    else:
        shown_features = features_shown

    node_means = np.array([get_means(node).reshape(-1) for node in nodes])
    node_vars = np.array([get_variances(node).reshape(-1) for node in nodes])
    node_stds = np.sqrt(node_vars)
    names = np.arange(1,len(nodes)+1,1)
    strength_separation = cluster_anova(spn)
    node_var, node_mean = cluster_mean_var_distance(nodes, spn)
    all_seps = {i: separation for i, separation in zip(shown_features, strength_separation)}
    for i in shown_features:
        if i not in categoricals:
            description_string = ''
            plot = p.plot_error_bar(names, node_means[:,i], node_vars[:,i], feature_names[i])
            strength = ['weak', 'moderate', 'strong', 'very strong', 'perfect']
            strength_values = [0.3, 0.6, 0.8, 0.99]
            strength_adv = strength[threshold(strength_values, strength_separation[i])]+'ly'
            var_outliers = np.where(node_var[:,i] > variance_threshold)[0]
            if len(var_outliers) == 1:
                node_string = ', '.join([str(v) for v in var_outliers])
                description_string += 'The variance of node {} is significantly larger then the average node. '.format(node_string)
            elif len(var_outliers) > 0:
                node_string = ', '.join([str(v) for v in var_outliers])
                description_string += 'The variances of the nodes {} are significantly larger then the average node. '.format(node_string)
            mean_high_outliers = np.where(node_mean[:,i] > mean_threshold)[0]
            mean_low_outliers = np.where(node_mean[:,i] < -mean_threshold)[0]
            if len(mean_high_outliers) == 1:
                node_string = ', '.join([str(v) for v in mean_high_outliers])
                description_string += 'The mean of node {} is significantly larger then the average node. '.format(node_string)
            elif len(mean_high_outliers) > 0:
                node_string = ', '.join([str(v) for v in mean_high_outliers])
                description_string += 'The means of the nodes {} are significantly larger then the average node. '.format(node_string)
            if len(mean_low_outliers) == 1:
                node_string = ', '.join([str(v) for v in mean_low_outliers])
                description_string += 'The mean of node {} is significantly smaller then the average node.'.format(node_string)
            elif len(mean_low_outliers) > 0:
                node_string = ', '.join([str(v) for v in mean_low_outliers])
                description_string += 'The means of the nodes {} are significantly smaller then the average node.'.format(node_string)
            if description_string or strength_separation[i] > separation_threshold:
                description_string = 'The feature "{}" is {} separated by the clustering. '.format(feature_names[i], strength_adv) + description_string
                iplot(plot)
                printmd(description_string)
    return all_seps


def node_categorical_description(spn, dictionary):
    context = dictionary['context']
    categoricals = context.get_categoricals()
    feature_names = context.feature_names

    enc = [dictionary['features'][cat]['encoder'] for cat in categoricals]
    summarized, contributions = categorical_nodes_description(spn, context)

    for i, cat in enumerate(categoricals):
        printmd('#### Distribution of {}'.format(feature_names[cat]))
        for cat_instance in [int(c) for c in context.get_domains_by_scope([cat])[0]]:
            name = enc[i].inverse_transform(cat_instance)
            contrib_nodes = summarized[cat]['contrib'][cat_instance][0]
            prop_of_instance = summarized[cat]['explained'][cat_instance][cat_instance]
            prop_of_nodes = prop_of_instance / np.sum(
                summarized[cat]['explained'][cat_instance])
            if prop_of_instance < 0.7:
                printmd('The feature "{}" is not separated well along the primary\
                        clusters.'.format(feature_names[cat]))
                break
            else:
                desc = '{}% of "{}" is captured by the nodes {}. The probability of\
                        "{}" for this group of nodes is {}%'
                printmd(desc.format(np.round(prop_of_instance * 100, 2),
                                    name,
                                    ', '.join([str(n) for n in contrib_nodes]),
                                    name,
                                    np.round(prop_of_nodes * 100, 2), ))


def classification(spn, numerical_data, dictionary):
    context = dictionary['context']
    categoricals = context.get_categoricals()
    misclassified = {}
    data_dict = {}
    for i in categoricals:
        y_true = numerical_data[:, i].reshape(-1, 1)
        query = np.copy(numerical_data)
        y_pred = predict_mpe(spn, i, query, context).reshape(-1, 1)
        misclassified[i] = np.where(y_true != y_pred)[0]
        misclassified_instances = misclassified[i].shape[0]
        data_dict[i] = np.concatenate((query[:, :i], y_pred, query[:, i+1:]), axis=1)
        printmd('For feature "{}" the SPN misclassifies {} instances, resulting in a precision of {}%.'.format(
                context.feature_names[i], misclassified_instances, np.round(100 * (1 - misclassified_instances/len(numerical_data)),2)))
    return misclassified, data_dict


def describe_misclassified(spn, dictionary, misclassified, data_dict,
                           numerical_data):
    context = dictionary['context']
    categoricals = context.get_categoricals()
    empty = np.array([[np.nan] * len(spn.scope)])
    for i in categoricals:
        if use_shapley:
            raise NotImplementedError
        else:
            if misclassified_explanations == 'all':
                show_misclassified = misclassified[i]
            elif isinstance(misclassified_explanations, int):
                num_choices = min(misclassified_explanations,
                                  len(misclassified[i]))
                show_misclassified = random.sample(misclassified[i].tolist(),
                                                   k=num_choices)
            else:
                show_misclassified = misclassified_explanations
            for inst_num in show_misclassified:
                instance = data_dict[i][inst_num:inst_num + 1]
                evidence = instance.copy()
                evidence[:, i] = np.nan
                prior = log_likelihood(spn, evidence)
                posterior = log_likelihood(spn, instance)
                total = 0
                all_nodes = []
                for j, node in enumerate(spn.children):
                    node_prob = np.exp(np.log(spn.weights[j]) + log_likelihood(spn, instance) - posterior)
                    total += node_prob
                    all_nodes.append((node_prob, j))
                all_nodes.sort()
                all_nodes.reverse()
                needed_nodes = []
                all_reps = []
                total_prob = 0
                for prob, idx in all_nodes:
                    node = Copy(spn.children[idx])
                    assign_ids(node)
                    total_prob += prob
                    needed_nodes.append(idx)
                    all_reps.append(mpe(node, empty)[0])
                    if total_prob > 0.9:
                        break
                real_value = dictionary['features'][i][
                    'encoder'].inverse_transform(
                    int(numerical_data[inst_num, i]))
                pred_value = dictionary['features'][i][
                    'encoder'].inverse_transform(
                    int(data_dict[i][inst_num, i]))
                printmd(
                    'Instance {} was predicted as "{}", even though it is "{}", because it was most similar to the following clusters: {}'.format(
                        inst_num, pred_value, real_value,
                        ', '.join(map(str, needed_nodes))))
                all_reps = np.array(all_reps).reshape(len(needed_nodes),
                                                      len(spn.scope))
                table = np.round(np.concatenate([instance, all_reps], axis=0),
                                 2)
                node_nums = np.array(['instance'] + needed_nodes).reshape(-1,
                                                                          1)
                table = np.append(node_nums, table, axis=1)

                iplot(p.plot_table([''] + context.feature_names, table.transpose()))


def explanation_vector_description(spn, dictionary, data_dict, cat_features):
    context = dictionary['context']
    categoricals = context.get_categoricals()
    num_features = len(spn.scope)
    feature_types = context.parametric_types
    domains = context.get_domains_by_scope(spn.scope)
    feature_names = context.feature_names
    all_combinations = list(itertools.product(categoricals, list(range(num_features))))
    if explanation_vectors_show == 'all':
        shown_combinations = all_combinations
    elif isinstance(explanation_vectors_show, int):
        num_choices = min(explanation_vectors_show, len(all_combinations))
        shown_combinations = random.sample(all_combinations, k=num_choices)
    else:
        shown_combinations = features_shown

    if explanation_vector_classes:
        shown_classes = explanation_vector_classes
    else:
        shown_classes = categoricals

    def plot_query(query, data, query_dict):
        if len(query[0]) == 0:
            return None
        conditional_evidence = np.full((1, num_features), np.nan)
        conditional_evidence[:, i] = data[0,i]
        gradients = conditional_gradient(spn, conditional_evidence, data[query])
        gradients_norm = np.linalg.norm(gradients, axis = 1).reshape(-1,1)
        _gradients = (gradients/gradients_norm)[:,k]
        discretize = np.histogram(_gradients, range=(-1,1), bins = 20)
        binsize = discretize[1][1] - discretize[1][0]
        if np.abs(_gradients.mean()) < explanation_vector_threshold:
            return _gradients
        header, description, plot = explanation_vector(_gradients, discretize, data, query, query_dict)
        if not header:
            return _gradients
        printmd(header)
        iplot(plot)
        printmd(description)
        return _gradients

    all_gradients = {}
    for i in shown_classes:
        all_gradients[i] = {}
        for j in domains[i]:
            all_gradients[i][j] = {}
            printmd('#### Class "{}": "{}"'.format(
                feature_names[i],
                dictionary['features'][i]['encoder'].inverse_transform(int(j))))
            test_query = np.where((data_dict[i][:,i] == j))
            if len(test_query[0]) == 0:
                printmd('For this particular class instance, no instances in the predicted data were found. \
                This might be because the predictive precision of the network was not high enough.')
                continue
            for k in range(num_features - 1):
                all_gradients[i][j][k] = {}
                this_range = [x for x in range(num_features) if x != i]
                instance = this_range[k]
                if (i,k) not in shown_combinations:
                    continue
                if instance in categoricals:
                    plot_data = []
                    for l in domains[instance]:
                        query = np.where((data_dict[i][:,i] == j) & (data_dict[i][:,instance] == l))
                        query_dict = {'type': 'categorical',
                                      'class': feature_names[i],
                                      'class_instance':
                                          dictionary['features'][i][
                                              'encoder'].inverse_transform(
                                              int(j)),
                                      'feature': feature_names[instance],
                                      'feature_instance':
                                          dictionary['features'][instance][
                                              'encoder'].inverse_transform(
                                              int(l)), 'feature_idx': instance,
                                      'class_idx': i}

                        gradients = f.gradient(spn, data_dict[i][query], i)
                        gradients_norm = np.linalg.norm(gradients, axis = 1).reshape(-1,1)
                        _gradients = (gradients/gradients_norm)[:,k]
                        discretize = np.histogram(_gradients, range=(-1,1), bins = 10)
                        binsize = discretize[1][1] - discretize[1][0]
                        plot_data.append((_gradients, discretize, query_dict['feature_instance']))
                    plot = p.plot_cat_explanation_vector(plot_data)
                    header = '##### Predictive categorical feature "{}": "{}"\n\n'.format(
                        query_dict['feature'], query_dict['feature_instance'])
                    printmd(header)
                    iplot(plot)

                    if _gradients is None:
                        all_gradients[i][j][k][l] = 0
                    else:
                        all_gradients[i][j][k][l] = _gradients.mean()
                else:
                    query = np.where((data_dict[i][:,i] == j))
                    query_dict = {'type': feature_types[instance],
                                  'class': feature_names[i],
                                  'class_instance': dictionary['features'][i][
                                      'encoder'].inverse_transform(int(j)),
                                  'feature': feature_names[instance],
                                  'feature_instance': '',
                                  'feature_idx': instance, 'class_idx': i}
                    _gradients = plot_query(query, data_dict[i], query_dict)
                    if _gradients is None:
                        all_gradients[i][j][k] = 0
                    else:
                        all_gradients[i][j][k] = _gradients.mean()
    return all_gradients


def explanation_vector(gradients, discretize, data, query, query_dict):
    '''
    Generates a textual description of an array of gradients. It describes general orientation and impact of the feature on a classification.
    Args:
        gradients (np.array): a numpy 1d array of gradient information
        discretize (np.array): a numpy 1d array of the binned gradients
        data (np.array): a numpy 2d array containing the original data on which the gradients were computed
        query (tuple of np.array): the entries of the array that were used to compute the gradients
        query_dict (dict): dictionary containing the class, feature and instance names and information about the feature type
    Returns:
        string: textual description of the gradient vector
    '''
    # general information about the direction of the gradients
    query_data = data[query]
    if np.abs(gradients.mean()) < explanation_vector_threshold:
        return None, None, None

    description_tree = expl_vec_grammar.ExplanationVectorDescription()
    description_tree.add_type(query_dict['type'])
    description_tree.add_feature(query_dict['feature'], query_dict['feature_idx'], query_dict['feature_instance'])
    description_tree.add_class(query_dict['class'], query_dict['class_idx'], query_dict['class_instance'])
    description_tree.add_data(query_data)
    description_tree.add_gradients(gradients)
    description_tree.build_description()
    data = p.plot_explanation_vectors(gradients, discretize)
    if query_dict['type'] == 'categorical':
        header = '##### Predictive categorical feature "{}": "{}"\n\n'.format(
                query_dict['feature'], query_dict['feature_instance'])
    else:
        header = '##### Predictive {} feature "{}"\n\n'.format(
                query_dict['type'], query_dict['feature'])
    return header, description_tree.get_text(), data

# ------------------------------------------------ #
# ------------------- OLD CODE ------------------- #
# ------------------------------------------------ #












def explain_misclassified(spn, dictionary, misclassified, categorical, predicted, original):
    k = categorical
    keys = misclassified.keys()
    root = spn.root
    for i, d in enumerate(misclassified[k]):
        predicted_nodes = f.prediction_nodes(spn, predicted[d:d+1], k)
        sorted_idx = np.argsort(predicted_nodes, axis = 0)
        total = 0
        expl = []
        for x in np.flip(sorted_idx, axis = 0):
            total += predicted_nodes[x]
            expl.append(x)
            if total > 0.9:
                break
        expl = np.array(expl).reshape(-1)
        proba_summed = np.sum(predicted_nodes[expl])
        prob_pred = 0
        prob_true = 0
        weights = np.sum([root.weights[i] for i in expl])
        for j in np.array(expl).reshape(-1):
            spn.root = root.children[j]
            prob_pred += np.exp(root.log_weights[j] + spn.marginalize([k]).eval(predicted[d:d+1]))[0]
            prob_true += np.exp(root.log_weights[j] + spn.marginalize([k]).eval(original[d:d+1]))[0]
        prob_pred /= weights
        prob_true /= weights

        printmd('Instance {} is best predicted by nodes {} ({}%) which have a probability for class "{}" of {}%, while for class "{}" the probability is {}%.'.format(
                d,
                expl,
                np.round(proba_summed*100, 2),
                dictionary['features'][k]['encoder'].inverse_transform(int(predicted[d,k])),
                np.round(prob_pred*100, 2),
                dictionary['features'][k]['encoder'].inverse_transform(int(original[d,k])),
                np.round(prob_true*100, 2)))
        spn.root = root
        feature_contib = f.get_feature_decomposition(spn, k)
        printmd()




def node_correlation(spn, dictionary):
    all_nodes = list(i for i, node in enumerate(spn.root.children) if get_spn_depth(node) > 1)

    if nodes == 'all':
        shown_nodes = all_nodes
    elif isinstance(nodes, int):
        num_choices = min(nodes, len(all_nodes))
        shown_nodes = random.sample(all_nodes, k=num_choices)
    else:
        shown_nodes = nodes

    root = spn.root
    shown_nodes = [spn.root.children[i] for i in shown_nodes]
    node_descritions = get_node_description(spn, spn.root, len(spn.scope))
    used_descriptions = [node_descritions['nodes'][i] for i, _ in enumerate(shown_nodes)]
    for i, (node, d) in enumerate(zip(shown_nodes, used_descriptions)):
        if not d['quick'] == 'shallow':
            printmd('### Correlations for node {}'.format(i))
            correlation_description(node, dictionary)


def print_conclusion(spn, dictionary, corr, nodes, node_separations, explanation_vectors):
    context = dictionary['context']
    feature_names = context.feature_names
    printmd('This concludes the automated report on the {} dataset.'.format(strip_dataset_name(spn.name)))

    correlated = np.where(np.abs(corr) > correlation_threshold)
    correlated_features = [(i,j) for i,j in zip(correlated[0], correlated[1]) if i > j]
    correlated_names = [(feature_names[i], feature_names[j]) for i, j in correlated_features]

    printmd('The initial findings show, that the following variables have a significant connections with each other.')

    printmd('\n'.join(['- "{}" - "{}"'.format(pair[0], pair[1]) for pair in correlated_names]))

    printmd('The intial clustering performed by the algorithm seperates the following features well:')

    separated_well = ['- ' + feature_names[i] for i in node_separations if node_separations[i] > 0.6]
    printmd('\n'.join(separated_well))

    relevant_classifications = []
    try:
        for cat in explanation_vectors:
            for cat_value in explanation_vectors[cat]:
                for predictor in explanation_vectors[cat][cat_value]:
                    if isinstance(explanation_vectors[cat][cat_value][predictor], dict):
                        summed = sum([abs(i) for i in explanation_vectors[cat][cat_value][predictor].values()]) / len(explanation_vectors[cat][cat_value][predictor])
                    else:
                        summed = abs(explanation_vectors[cat][cat_value][predictor])
                    if summed > explanation_vector_threshold:
                        encoder_cat = dictionary['features'][cat]['encoder']
                        categorical = '{} - {}'.format(feature_names[cat],
                                encoder_cat.inverse_transform(cat_value))
                        class_descriptor = str(feature_names[cat]) + ' - ' + str(encoder_cat.inverse_transform(cat_value))
                        relevant_classifications.append((class_descriptor, feature_names[predictor]))
    except Exception as e:
        pass

    explanation_summary = [fix_sentence(generate_from_file('.', 'conclusion_explanation_vector.txt')[1].raw_str).format(x=x,y=y) for x,y in relevant_classifications]
    printmd(' '.join(explanation_summary))

    printmd('''If you want to explore the dataset further, you can use the interactive notebook to add your own queries to the network.
    ''')