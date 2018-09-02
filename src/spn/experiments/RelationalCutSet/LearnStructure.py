'''
Created on August 16, 2018

@author: Alejandro Molina
'''

import glob

import pickle
from collections import OrderedDict

import numpy as np
import os

from lark import Tree
from sklearn.metrics import confusion_matrix
from tqdm._tqdm import tqdm

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Marginalization import marginalize
from spn.algorithms.Validity import is_valid
from spn.io.Text import spn_to_str_ref_graph
from spn.structure.Base import Sum, Product, Node, assign_ids, rebuild_scopes_bottom_up, get_nodes_by_type
from spn.structure.leaves.parametric.Parametric import CategoricalDictionary
import itertools


class Dependency:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parents = []

    @staticmethod
    def parse(txt):
        from lark import Lark
        grammar = r"""
                    %import common.WS
                    %ignore WS
                    %import common.WORD -> WORD
                    %import common.CNAME   -> STRING
                    TABLE.1: "@" STRING
                    ATTRIBUTE.2: STRING
                    ?parent_node.3: node "(" node ")"
                    ?node_list.4: node ("," node)+
                    ?node: TABLE | ATTRIBUTE
                        | node_list
                        | parent_node
                    """
        tree = Lark(grammar, start='node').parse(txt)

        def parse_tree(tree, parents=None):
            result = None
            if type(tree) == Tree:
                if tree.data == "node_list":
                    result = []
                    for c in tree.children:
                        result.extend(parse_tree(c, parents=parents))
                    return result
                elif tree.data == "parent_node":
                    result = Dependency(str(tree.children[0]))
                    for c in tree.children[1:]:
                        child = parse_tree(c, parents=parents + [result])
                        result.children.extend(child)
            else:
                result = Dependency(str(tree))

            if parents is not None:
                result.parents = parents

            return [result]

        return parse_tree(tree, [])[0]

    def __repr__(self):
        txt = ("%s %s" % (self.name, ",".join(map(str, self.children)))).strip()
        return "(%s)" % txt


def parse_attributes(path):
    attributes_in_table = {}
    scopes = {}
    meta_data = {}
    attribute_owners = {}
    with open(path + "attributes.csv", "r") as attfile:
        for line in attfile:
            table = line.strip().split(':')
            table_name = table[0]
            key_attributes = table[1].split(',')
            for ka in key_attributes:
                if ka[-1] == "*":
                    attribute_owners[ka[:-1]] = table_name

            attributes = table[1].replace("*", "").split(',')
            meta_data[table_name] = {key: index for index, key in enumerate(attributes)}

            for att in attributes:
                if att not in scopes:
                    scopes[att] = len(scopes)

                if att not in attributes_in_table:
                    attributes_in_table[att] = set()
                attributes_in_table[att].add(table_name)
    return (attributes_in_table, scopes, meta_data, attribute_owners)


def load_tables(path, meta_data, debug=False):
    tables = {}
    for fname in glob.glob(path + "*.tbl"):
        table_name = os.path.splitext(os.path.basename(fname))[0]
        print("loading", table_name, "from", fname)
        table = np.genfromtxt(fname, delimiter='|')[:, 0:len(meta_data[table_name])]
        tables[table_name] = table
        assert not np.any(np.isnan(table)), "found missing values in table: " + table_name

    # if in debug mode, reduce size
    if debug:
        tables["Ratings"] = tables["Ratings"][0:25]

        cond = np.zeros(tables["Users"].shape[0])
        for ruid in tables["Ratings"][:, 0]:
            cond = np.logical_or(cond, tables["Users"][:, 0] == ruid)
        tables["Users"] = tables["Users"][cond, :]

        cond = np.zeros(tables["Movies"].shape[0])
        for ruid in tables["Ratings"][:, 1]:
            cond = np.logical_or(cond, tables["Movies"][:, 0] == ruid)
        tables["Movies"] = tables["Movies"][cond, :]

    return tables


def get_keys(dep_tree, meta_data, attributes_in_table):
    # keys are the attributes that show up in more than one table.
    keys = set()
    keys_per_table = {}
    for att, table_names in attributes_in_table.items():
        if len(table_names) > 1:
            keys.add(att)
            for table_name in table_names:
                if table_name not in keys_per_table:
                    keys_per_table[table_name] = []
                keys_per_table[table_name].append(att)

    keys_per_attribute = {}
    ancestors = {}

    def process_dep_tree(dep_node):
        att_name = dep_node.name

        ancestors[att_name] = set(list(map(lambda d: d.name, dep_node.parents)))
        if att_name[0] == '@':
            keys_per_attribute[att_name] = keys_per_table[att_name[1:]]
            return

        tables = attributes_in_table[att_name]

        if len(dep_node.parents) == 0:
            # i'm root, I have no parents
            keys_per_attribute[att_name] = []
        elif len(tables) == 1:
            # i belong to only one table
            table = [t for t in tables][0]
            table_atts = meta_data[table]
            keys_per_attribute[att_name] = [att for att in ancestors[att_name] if att in table_atts]
        else:
            # i belong to multiple tables, use the one that has the attribute with my parent
            table_atts = meta_data[[t for t in tables if dep_node.parents[-1].name in meta_data[t]][0]]
            keys_per_attribute[att_name] = [att for att in ancestors[att_name] if att in table_atts]

        for c in dep_node.children:
            process_dep_tree(c)

    process_dep_tree(dep_tree)
    return keys, keys_per_attribute, ancestors


def build_cache(tables, meta_data, table_keys, scopes, attribute_owners):
    def factorize_data(lower, higher, table, non_key_features):
        table = table[lower:higher]

        for (att, pos) in non_key_features:
            pdf_v, pdf_c = np.unique(table[:, pos], return_counts=True)
            pchild = CategoricalDictionary(p=OrderedDict(zip(pdf_v, pdf_c)), scope=scopes[att])
            pchild.att = att
            pchild.debug = lambda self: "C_%s(%s,%s)" % (self.id, self.params, self.att)
            yield pchild

    def process_data(table_name, lower, higher, table, table_meta_data, scopes, keys_left, non_key_features, siblings,
                     cache):
        # dig into the constraints
        curr_att = keys_left[0]
        att_pos = table_meta_data[curr_att]

        constraint_table = cache.get(curr_att, None)
        if constraint_table is None:
            cache[curr_att] = constraint_table = {}
        else:
            assert False

        table = table[lower:higher]
        column = table[:, att_pos]
        vals, counts = np.unique(column, return_counts=True)
        for val, count in zip(vals, counts):
            l = np.searchsorted(column, val, side='left')
            h = np.searchsorted(column, val, side='right')

            node = None
            if attribute_owners[curr_att] == table_name and False:
                # if I'm the owner, we add the filter
                node = CategoricalDictionary(p={val: count}, scope=scopes[curr_att])
                node.att = curr_att
                node.debug = lambda self: "C_%s(%s,%s)" % (self.id, self.params, self.att)

            if len(keys_left) > 1:
                val_constraint_table = constraint_table.get(val, None)
                if val_constraint_table is None:
                    constraint_table[val] = val_constraint_table = {}
                else:
                    assert False

                new_siblings = list(siblings)
                if node is not None:
                    new_siblings += [node]
                process_data(table_name, l, h, table, table_meta_data, scopes, keys_left[1:], non_key_features,
                             new_siblings, val_constraint_table)
            else:
                p_node = Product()
                # p_node.debug = lambda self: "P_%s(%s)" % (self.id, ",".join([str(p) for p in self.children]))

                if val not in constraint_table:
                    constraint_table[val] = (p_node, count)
                else:
                    assert False

                p_node.children.extend(siblings)
                if node is not None:
                    p_node.children.append(node)
                p_node.children.extend(factorize_data(l, h, table, non_key_features))

    cache = {}
    for table_name, constraint_groups in table_keys.items():
        table = tables[table_name]
        print("loading: ", table_name)
        table_meta_data = meta_data[table_name]

        table_cache = cache.get(table_name, None)
        if table_cache is None:
            table_cache = cache[table_name] = {}

        for keys in constraint_groups:  # for different clustering orders
            constraint_atts = None

            print("loading constraint group: ", constraint_groups)
            table_key = tuple(keys)
            constraint_cache = table_cache.get(table_key, None)
            if constraint_cache is None:
                constraint_cache = table_cache[table_key] = {}

            sort_order = [table[:, table_meta_data[att_name]] for att_name in reversed(keys)]
            sorted_table = table[np.lexsort(sort_order), :].astype(float)

            non_key_features = [(k, table_meta_data[k]) for k in table_meta_data.keys() if
                                k not in keys and (k not in attribute_owners or attribute_owners[k] == table_name)]

            process_data(table_name, 0, sorted_table.shape[0], sorted_table, table_meta_data, scopes, keys,
                         non_key_features, [],
                         constraint_cache)

    return cache


def get_table_ancestors(dep_tree, attributes_in_table):
    keys = set()
    keys_per_table = {}
    for att, table_names in attributes_in_table.items():
        if len(table_names) > 1:
            keys.add(att)
            for table_name in table_names:
                if table_name not in keys_per_table:
                    keys_per_table[table_name] = []
                keys_per_table[table_name].append(att)

    table_keys = {}
    ancestors = {}

    def process_dep_tree(dep_node):
        att_name = dep_node.name

        if att_name[0] == '@':
            table_name = att_name[1:]
            all_keys_in_table = set(keys_per_table[table_name])
            ancestors_per_table = [p.name for p in dep_node.parents]
            ancestors[table_name] = ancestors_per_table
            table_keys[table_name] = [[a for a in ancestors_per_table if a in all_keys_in_table]]

        for c in dep_node.children:
            process_dep_tree(c)

    process_dep_tree(dep_tree)

    keys_to_tables = {}
    for table_name, groups in table_keys.items():
        for group in groups:
            key = tuple(group)
            if key not in keys_to_tables:
                keys_to_tables[key] = []
            keys_to_tables[key].append(table_name)

    return table_keys, keys_to_tables, ancestors


def get_dependncy_keys(dep_tree, table_keys, attribute_owners, path_constraints):
    children_tables = [dc for dc in dep_tree.children if dc.name[0] == "@"]
    # if my children contains a table, I'm a key, otherwise traverse down.
    if len(children_tables) == 0:
        # we have to traverse down
        assert len(dep_tree.children) == 1
        return get_dependncy_keys(dep_tree.children[0], table_keys, attribute_owners, path_constraints)

    path = [d.name for d in dep_tree.parents]
    path.append(dep_tree.name)

    keys = []
    for dc in children_tables:
        table_name = dc.name[1:]
        for key in table_keys[table_name]:
            if key == path[-len(key):]:
                keys.append((table_name, tuple(key)))
                break
    assert len(keys) > 0, "invalid path, no table has keys :" + str(path)

    return [(keys, dep_tree)]


def get_constraint_values(table_names_keys, path_constraints, cache):
    def iterate_all_values(table_names_keys, path_constraints, cache):
        for table_name, key in table_names_keys:
            cache_values = cache[table_name][key]
            key_constraints = set(key)
            filtered_path_constraints = [pc for pc in path_constraints if pc[0] in key_constraints]
            hierarchy_path_constraints = [pc for pc in path_constraints if pc[0] not in key_constraints]

            def traverse_cache_values(hierarchy_p_c, cache_vals, path_constraints):
                if len(path_constraints) > 0:
                    constraint = path_constraints[0]
                    for c in traverse_cache_values(hierarchy_p_c, cache_vals[constraint[0]][constraint[1]],
                                                   path_constraints[1:]):
                        yield ([constraint] + c[0], c[1], c[2], c[3])
                else:
                    # { key: {val : key ...
                    for k, pointer in cache_vals.items():
                        # { key: {val
                        for value, sub_pointer in pointer.items():
                            if isinstance(sub_pointer, dict):
                                for constraint in traverse_cache_values(hierarchy_p_c, sub_pointer, path_constraints):
                                    result = [(k, value)]
                                    result.extend(constraint[0])
                                    yield (result, constraint[1], constraint[2], constraint[3])
                            else:
                                yield ([(k, value)], sub_pointer[0], sub_pointer[1], hierarchy_p_c)

            yield from traverse_cache_values(hierarchy_path_constraints, cache_values, filtered_path_constraints)

    result = {}
    for constraint, node, count, hierarchy_path_constraints in iterate_all_values(table_names_keys, path_constraints,
                                                                                  cache):

        constraint = tuple(hierarchy_path_constraints + constraint)
        if constraint not in result:
            result[constraint] = []
        result[constraint].append((node, count))

    for rk, rv in result.items():
        if len(rv) == len(table_names_keys):
            yield (rk, rv)
        else:
            print("missing", rk, rv)


def build_spn(dep_tree, table_keys, scopes, attribute_owners, path_constraints=None, cache=None):
    def build_recursive(dep_tree, table_keys, scopes, attribute_owners, path_constraints=None, cache=None):
        if path_constraints is None:
            path_constraints = []

        new_node = Sum()
        for (table_names_keys, dep_node) in get_dependncy_keys(dep_tree, table_keys, attribute_owners,
                                                               path_constraints):

            for constraint_configuration, cached_node_count in get_constraint_values(table_names_keys, path_constraints,
                                                                                     cache):
                p_node = Product()
                new_node.children.append(p_node)
                count_value = 1

                for cached_node, node_count in cached_node_count:
                    p_node.children.append(cached_node)
                    count_value *= node_count

                for dep_children_node in dep_node.children:
                    if dep_children_node.name[0] == '@':
                        continue

                    node, count = build_recursive(dep_children_node, table_keys, scopes, attribute_owners,
                                                  path_constraints=constraint_configuration,
                                                  cache=cache)
                    p_node.children.append(node)
                    count_value *= count
                new_node.weights.append(count_value)

        wsum = np.sum(new_node.weights)
        # new_node.weights = [w / wsum for w in new_node.weights]

        return new_node, wsum

    root, count = build_recursive(dep_tree, table_keys, scopes, attribute_owners, path_constraints=path_constraints,
                                  cache=cache)
    if True:
        for sum_node in get_nodes_by_type(root, Sum):
            normalization = np.sum(sum_node.weights)
            sum_node.weights = [w / normalization for w in sum_node.weights]
        for cat_node in get_nodes_by_type(root, CategoricalDictionary):
            psum = 0
            for name, count in cat_node.p.items():
                psum += count
            cat_node.p = {name: count / psum for name, count in cat_node.p.items()}
    return root


if __name__ == '__main__':
    path = "/Users/alejomc/Downloads/100k/"

    # path = "/Users/alejomc/PycharmProjects/idbspn/datasets/ml-tiny/"
    # path = "/Users/alejomc/PycharmProjects/idbspn/datasets/ml-small/"
    test_path = path + "test/"

    attributes_in_table_test, test_scopes, test_meta_data, _ = parse_attributes(test_path)
    test_tables = load_tables(test_path, test_meta_data, debug=False)

    with open(path + "dependencies.txt", "r") as depfile:
        dep_tree = Dependency.parse(depfile.read())
    print(dep_tree)

    cluster_by = {}
    with open(path + "cluster_by.txt", "r") as cbfile:
        for l in cbfile.readlines():
            table_name, cluster_by_atts = l.split(':')
            cluster_by[table_name] = cluster_by_atts.strip().split(',')

    attributes_in_table, scopes, meta_data, attribute_owners = parse_attributes(path)

    table_keys, keys_to_tables, ancestors = get_table_ancestors(dep_tree, attributes_in_table)

    tables = load_tables(path, meta_data, debug=False)

    spn = None

    file_cache_path = "/tmp/csn.bin"
    if not os.path.isfile(file_cache_path):
        cache = build_cache(tables, meta_data, table_keys, scopes, attribute_owners)

        spn = build_spn(dep_tree, table_keys, scopes, attribute_owners, path_constraints=None, cache=cache)
        rebuild_scopes_bottom_up(spn)
        print(spn)
        assign_ids(spn)
        val, msg = is_valid(spn)
        print(spn_to_str_ref_graph(spn))
        assert val, msg
        # 0/0
        # print(spn_to_str_ref_graph(spn))

        keep = set(scopes.values())
        keep.discard(scopes["userid"])
        keep.discard(scopes["movieid"])

        # with open(file_cache_path, 'wb') as f:
        #    pickle.dump(spn, f, pickle.HIGHEST_PROTOCOL)
    else:
        print("loading cached spn")
        with open(file_cache_path, 'rb') as f:
            spn = pickle.load(f)
        print("loaded cached spn")


    def to_data(scopes, rows, **kwargs):
        data = np.zeros((rows, max(scopes.values()) + 1))
        data[:] = np.nan
        for k, v in kwargs.items():
            data[:, scopes[k]] = v
        return data


    def compute_conditional(spn, rows, scopes, query, **evidence):
        q_e = dict(query)
        q_e.update(evidence)
        # query_str = ",".join(map(lambda t: "%s=%s" % (t[0], t[1]), query.items()))
        # evidence_str = ",".join(map(lambda t: "%s=%s" % (t[0], t[1]), evidence.items()))
        # prob_str = "P(%s|%s)" % (query_str, evidence_str)
        # print("computing ", prob_str)

        a = log_likelihood(spn, to_data(scopes, rows, **q_e), debug=True)
        # print("query ", query_str, np.exp(a))
        b = 0  # log_likelihood(spn, to_data(scopes, rows, **evidence), debug=True)
        # print("evidence ", evidence_str, b, np.exp(b))
        result = np.exp(a - b)
        # print(prob_str, "=", result, "query ", query_str, np.exp(a), evidence_str, np.exp(b))
        return result


    # print(spn_to_str_ref_graph(spn))

    evidence_atts = ['year', 'action', 'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary', 'drama',
                     'fantasy', 'filmnoir', 'horror', 'musical', 'mystery', 'romance', 'scifi',
                     'thriller', 'war', 'western', 'age', 'gender', 'occupation'] + ['genre_group1', 'genre_group2',
                                                                                     'genre_group3', 'genre_group4',
                                                                                     'genre_group5']

    test_table = test_tables["Test"]
    rating_values = np.zeros((test_table.shape[0], 5))
    for rating in tqdm(list(range(5))):
        evidence_query = {}
        for ea in evidence_atts:
            evidence_query[ea] = test_table[:, test_scopes[ea]]
        rating_values[:, rating] = compute_conditional(spn, test_table.shape[0], scopes,
                                                       {'rating': np.repeat(rating + 1, test_table.shape[0], axis=0)},
                                                       **evidence_query)[:, 0]
    pred_ratings = np.argmax(rating_values, axis=1) + 1
    true_ratings = test_table[:, test_scopes["rating"]]
    print(confusion_matrix(true_ratings, pred_ratings))

    for i in range(4):
        sdiff = np.sum(np.abs(true_ratings - pred_ratings) <= i)
        print("rating diffeerence at most", i, sdiff, sdiff / test_table.shape[0])

    from sklearn.metrics import mean_squared_error
    from math import sqrt


    def rmse(prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return sqrt(mean_squared_error(prediction, ground_truth))


    print("rmse", rmse(pred_ratings, true_ratings))

# compute_conditional(spn, scopes, {'rating': 5}, age=25.0, occupation=3.0)
# compute_conditional(spn, scopes, {'rating': 5}, fantasy=1.0, romance=1.0)
# compute_conditional(spn, scopes, {'rating': 5}, fantasy=1.0, romance=1.0, age=25.0)
# compute_conditional(spn, scopes, {'rating': 3}, fantasy=1.0, romance=1.0, age=25.0)
# compute_conditional(spn, scopes, {'rating': 3}, crime=1.0, occupation=4.0, age=25.0)
