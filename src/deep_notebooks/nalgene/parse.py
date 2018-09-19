import re
import math
import random
import json
import sys
from deep_notebooks.nalgene.node import *

SHIFT_WIDTH = 4

start_space = r'^(    )*'

def count_indent(s):
    indent = len(re.match(start_space, s).group(0))
    return math.floor(indent / SHIFT_WIDTH)

def parse_string(base_dir, string):
    lines = string.split('\n')
    lines = [line for line in lines if not re.match(r'^\s*#', line)]
    parsed = Node('')
    indexes = [-1]
    level = 0
    last_ind = 0

    for li in range(len(lines)):
        line = lines[li]
        ind = count_indent(line)
        line = re.sub(start_space, '', line).strip()
        if len(line) == 0: continue

        if level == 0 and line.startswith('@import'):
            filename = line.split(' ')[1]
            imported = parse_file(base_dir, filename)
            for child in imported:
                parsed.add(child)
                indexes[level] += 1
            continue

        if ind == last_ind: # Next item in a list
            indexes[level] += 1
        elif ind > last_ind: # Child item
            level += 1
            indexes.append(0)
        elif ind < last_ind: # Up to next item in parent list
            diff = (last_ind - ind)
            for i in range(last_ind - ind):
                level -= 1
                indexes.pop()
            indexes[level] += 1

        parsed.add_at(line, indexes)
        last_ind = ind

    return parsed

def tokenizeLeaf(n):
    n.type = 'seq'
    for s in n.key.split(' '):
        added = n.add(s)
    n.key = 'seq'

def parse_file(base_dir, filename):
    parsed = parse_string(base_dir, open(base_dir + '/' + filename).read())
    return parsed

def parse_dict(obj, obj_key='%'):
    tree = Node(obj_key)
    if isinstance(obj, dict):
        for key in obj.keys():
            tree.add(parse_dict(obj[key], key))
        return tree
    else:
        tree.add(obj)
        return tree
