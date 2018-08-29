'''
Created on August 16, 2018

@author: Alejandro Molina
'''

import glob

import pickle
import numpy as np
import os

from lark import Tree

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Marginalization import marginalize
from spn.algorithms.Validity import is_valid
from spn.io.Text import spn_to_str_ref_graph
from spn.structure.Base import Sum, Product, assign_ids, rebuild_scopes_bottom_up
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
                    TABLE.1: "@" WORD
                    ATTRIBUTE.2: WORD
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

            attributes = table[1].replace("*","").split(',')
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
        tables[table_name] = np.genfromtxt(fname, delimiter='|')[:,0:len(meta_data[table_name])]

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


def get_values(attributes_in_table, meta_data, tables, att_name, constraints, table_row_idxs=None):
    result_set = None
    value_count = {}
    filtered_table_row_idxs_by_value = {}
    for table_name in attributes_in_table[att_name]:
        table_meta_data = meta_data[table_name]
        att_pos = table_meta_data[att_name]
        table = tables[table_name]

        # this is optimization
        row_mask = None
        row_idx = None
        if table_row_idxs is not None and table_name in table_row_idxs:
            row_idx = table_row_idxs[table_name]
            row_mask = row_idx > -1
        if row_mask is None:
            row_mask = np.ones(table.shape[0]) == 1
            row_idx = np.array(list(range(table.shape[0])))

        # this computes the constraints
        for constraint_attribute, constraint_value in constraints.items():
            constraint_att_pos = table_meta_data.get(constraint_attribute, None)
            if constraint_att_pos is None:
                continue
            row_mask = row_mask & (table[row_idx, constraint_att_pos] == constraint_value)

        # data contains the column with all the values we want, after filtering
        data = table[row_idx[row_mask], att_pos]

        # obtain unique values
        unique, rev_idx, count = np.unique(data, return_counts=True, return_inverse=True)
        for i, val in enumerate(unique):

            # this is optimization, to not fully scan all tables always
            table_row_idx_by_val = filtered_table_row_idxs_by_value.get(val, None)
            if table_row_idx_by_val is None:
                table_row_idx_by_val = {}
                filtered_table_row_idxs_by_value[val] = table_row_idx_by_val
            table_row_idx = table_row_idx_by_val.get(table_name, None)
            if table_row_idx is None:
                table_row_idx_by_val[table_name] = row_idx[rev_idx == i]

            # how many instances were found on the join?
            if val not in value_count:
                value_count[val] = count[i]
            else:
                value_count[val] = max(value_count[val], count[i])

        # we are only interested in the values that are intersected on all tables
        table_values = set(unique)
        if result_set is None:
            result_set = table_values
        else:
            result_set.intersection_update(table_values)

    # return the values, the counts, and the row indexes where those values are found
    for val in result_set:
        yield (val, value_count[val], filtered_table_row_idxs_by_value[val])


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


def build_cache(tables, meta_data, table_keys, attribute_owners):
    cache = {}
    for table_name, constraint_groups in table_keys.items():
        table = tables[table_name]

        for keys in constraint_groups:  # for different clustering orders
            constraint_atts = None

            for att_name in keys:
                att_pos = meta_data[table_name][att_name]

                new_constraints = []
                for v in np.unique(table[:, att_pos]):
                    new_constraints.append((att_name, v))
                if constraint_atts is None:
                    constraint_atts = new_constraints
                else:
                    constraint_atts = list(itertools.product(constraint_atts, new_constraints))

            for constraint in constraint_atts:
                # get data
                idx = np.ones(table.shape[0]) == 1
                if not isinstance(constraint[0], tuple):
                    constraint = [constraint]

                for grounding in constraint:
                    idx = idx & (table[:, meta_data[table_name][grounding[0]]] == grounding[1])
                group_data = table[idx, :].astype(float)

                if group_data.shape[0] == 0:
                    print("WARNING: empty group data" )
                    continue

                # compute leaf
                if table_name not in cache:
                    cache[table_name] = {}
                table_cache = cache[table_name]
                table_key = tuple(keys)
                if table_key not in table_cache:
                    table_cache[table_key] = {}
                constraint_cache = table_cache[table_key]

                p_node = Product()
                for i, (c, val) in enumerate(constraint):
                    if c not in constraint_cache:
                        constraint_cache[c] = {}
                    constraint_cache = constraint_cache[c]
                    if val not in constraint_cache:
                        if i == len(constraint)-1:
                            constraint_cache[val] = (p_node, np.sum(idx))
                        else:
                            constraint_cache[val] = {}
                    constraint_cache = constraint_cache[val]

                constraint_vars = [c[0] for c in constraint]
                for att, pos in meta_data[table_name].items():
                    if att in constraint_vars:
                        continue
                    if att in attribute_owners and attribute_owners[att] != table_name:
                        continue
                    pdf_v, pdf_c = np.unique(group_data[:, pos], return_counts=True)
                    p_node.children.append(CategoricalDictionary(p=dict(zip(pdf_v, pdf_c / group_data.shape[0])),
                                                                 scope=scopes[att]))

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


def build_csn(attributes_in_table, meta_data, tables, dependency_node, table_keys, ancestors, constraints=None,
              table_row_idxs=None, cache=None,
              debug=False):
    if constraints is None:
        constraints = {}

    att_name = dependency_node.name

    if att_name[0] == "@":
        key = keys_per_attribute[att_name]
        cache_key = tuple([(k, constraints[k]) for k in key if k in constraints])
        return cache[att_name[1:]][cache_key]

    new_node = Sum()

    new_constraints = dict(constraints)

    for val, count, filtered_table_row_idxs in get_values(attributes_in_table, meta_data, tables, att_name,
                                                          constraints, table_row_idxs):
        if debug:
            print("att_name", att_name, val)
        new_node.weights.append(count)

        p_node = Product()
        p_node.children.append(CategoricalDictionary(p={float(val): 1.0}, scope=scopes[att_name]))

        for dep_node in dependency_node.children:
            new_constraints[att_name] = val
            p_node.children.append(
                build_csn(attributes_in_table, meta_data, tables, dep_node, table_keys, ancestors,
                          constraints=new_constraints,
                          table_row_idxs=filtered_table_row_idxs,
                          cache=cache,
                          debug=False))

        new_node.children.append(p_node)

    wsum = np.sum(new_node.weights)
    new_node.weights = [w / wsum for w in new_node.weights]

    if can_cache:
        cache[cache_key] = new_node

    return new_node

def get_dependncy_keys(dep_tree, table_keys, path_constraints):
    children_tables = [dc for dc in dep_tree.children if dc.name[0] == "@"]
    #if my children contains a table, I'm a key, otherwise traverse down.
    #assert len(children_tables) == 1

    path = [d.name for d in dep_tree.parents]
    path.append(dep_tree.name)

    keys = []
    for dc in children_tables:
        table_name = dc.name[1:]
        for key in table_keys[table_name]:
            if key == path[-len(key):]:
                keys.append((table_name, tuple(key), dep_tree))
                break


    return [keys[0]]

def get_constraint_values(key, table_name, path_constraints, cache):
    cache_values = cache[table_name][key]

    for k in key:
        constraints = [c for c in path_constraints if c[0] == k]
        if len(constraints) > 0:
            cache_values = cache_values[k][constraints[0][1]]
            continue
        else:
            cache_values = cache_values[k]

    for k, v in cache_values.items():
        yield (k, v[1])


def build_csn2(dep_tree, table_keys, scopes, path_constraints=None, cache=None):
    if path_constraints is None:
        path_constraints = []

    if dep_tree.name[0] == "@":
        table_name = dep_tree.name[1:]
        path_vars = [p[0] for p in path_constraints]
        matching_keys = [k for k in table_keys[table_name] if k == path_vars[-len(k):]][0]
        cache_nodes = cache[table_name][tuple(matching_keys)]

        for pc in path_constraints[-len(matching_keys):]:
            cache_nodes = cache_nodes[pc[0]][pc[1]]

        return cache_nodes[0]

    new_node = Sum()
    for table_name, key, dep_node in get_dependncy_keys(dep_tree, table_keys, path_constraints):
        new_constraints = []
        new_constraints.extend(path_constraints)
        new_constraints.append(None)


        constraint_name = dep_node.name
        for constraint_value, count in get_constraint_values(key, table_name, path_constraints, cache):
            p_node = Product()
            new_node.children.append(p_node)
            new_node.weights.append(count)

            p_node.children.append(CategoricalDictionary(p={float(constraint_value): 1.0}, scope=scopes[constraint_name]))

            new_constraints[-1] = (constraint_name, constraint_value)
            for dep_children_node in dep_node.children:
                p_node.children.append(build_csn2(dep_children_node, table_keys, scopes, path_constraints=new_constraints, cache=cache))


        wsum = np.sum(new_node.weights)
    new_node.weights = [w / wsum for w in new_node.weights]

    return new_node



if __name__ == '__main__':
    path = "/Users/alejomc/Downloads/100k/"
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

    cache = build_cache(tables, meta_data, table_keys, attribute_owners)

    spn = None

    file_cache_path = "/tmp/csn.bin"
    if not os.path.isfile(file_cache_path) or True:
        spn = build_csn2(dep_tree, table_keys, scopes, path_constraints=None, cache=cache)
        rebuild_scopes_bottom_up(spn)
        print(spn)
        assign_ids(spn)
        print(is_valid(spn))

        keep = set(scopes.values())
        keep.discard(scopes["userid"])
        keep.discard(scopes["movieid"])

        marg = marginalize(spn, keep)
        with open(file_cache_path, 'wb') as f:
            pickle.dump((spn, marg), f, pickle.HIGHEST_PROTOCOL)
    else:
        print("loading cached spn")
        with open(file_cache_path, 'rb') as f:
            spn, marg = pickle.load(f)
        print("loaded cached spn")


    def to_data(scopes, **kwargs):
        data = np.zeros((1, max(scopes.values()) + 1))
        data[:] = np.nan
        for k, v in kwargs.items():
            data[0, scopes[k]] = v
        return data


    def compute_conditional(spn, scopes, query, **evidence):
        q_e = dict(query)
        q_e.update(evidence)
        query_str = ",".join(map(lambda t: "%s=%s" % (t[0], t[1]), query.items()))
        evidence_str = ",".join(map(lambda t: "%s=%s" % (t[0], t[1]), evidence.items()))
        prob_str = "P(%s|%s)" % (query_str, evidence_str)
        # print("computing ", prob_str)

        a = log_likelihood(spn, to_data(scopes, **q_e), debug=False)
        # print("query ", query_str, np.exp(a))
        b = log_likelihood(spn, to_data(scopes, **evidence), debug=False)
        # print("evidence ", evidence_str, b, np.exp(b))
        result = np.exp(a - b)
        print(prob_str, "=", result, "query ", query_str, np.exp(a), evidence_str, np.exp(b))
        return result


    print(spn_to_str_ref_graph(spn))





    compute_conditional(spn, scopes, {'rating': 5}, age=25.0, occupation=3.0)
    compute_conditional(spn, scopes, {'rating': 5}, fantasy=1.0, romance=1.0)
    compute_conditional(spn, scopes, {'rating': 5}, fantasy=1.0, romance=1.0, age=25.0)
    compute_conditional(spn, scopes, {'rating': 3}, fantasy=1.0, romance=1.0, age=25.0)
    compute_conditional(spn, scopes, {'rating': 3}, crime=1.0, occupation=4.0, age=25.0)
