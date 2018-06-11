'''
Created on June 10, 2018

@author: Alejandro Molina
'''
from spn.algorithms.Statistics import get_structure_stats
from spn.algorithms.Validity import is_valid
from spn.experiments.RandomSPNs.LearnRGSPN import Make_SPN_from_RegionGraph
from spn.experiments.RandomSPNs.region_graph import RegionGraph
import numpy as np

from spn.structure.Base import Sum, get_nodes_by_type, Leaf, Product
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support


def get_execution_layers(spn):
    all_nodes = set(get_nodes_by_type(spn, ntype=(Sum, Product)))
    filter_type = Product
    layers = [get_nodes_by_type(spn, Leaf)]
    layer_types = [Leaf]
    seen_nodes = set(layers[0])
    while len(all_nodes) > 0:
        filtered_nodes = []
        new_all_nodes = set()

        for n in all_nodes:
            if isinstance(n, filter_type) and set(n.children).issubset(seen_nodes):
                filtered_nodes.append(n)
            else:
                new_all_nodes.add(n)

        layer_types.append(filter_type)

        if filter_type == Product:
            filter_type = Sum
        else:
            filter_type = Product

        assert all_nodes == new_all_nodes | set(filtered_nodes)

        layers.append(filtered_nodes)
        seen_nodes.update(filtered_nodes)
        all_nodes = new_all_nodes

    return layers, layer_types

if __name__ == '__main__':
    rg = RegionGraph(range(28 * 28))
    #for _ in range(0, 2):
    for _ in range(0, 20):
        rg.random_split(2, 2)

    rg_layers = rg.make_layers()

    vector_list = Make_SPN_from_RegionGraph(rg_layers, np.random.RandomState(100),
                                            num_classes=10, num_gauss=20, num_sums=20)

    spns = vector_list[-1][0]

    tmp_root = Sum()
    tmp_root.children.extend(spns)

    layers, layer_types = get_execution_layers(tmp_root)

    for i, lt in enumerate(layer_types):
        print(lt, len(layers[i]))


    max_id = max(map(lambda n: n.id, get_nodes_by_type(tmp_root)))
    print(max_id)

    children_sizes = list(map(lambda n: len(n.children), get_nodes_by_type(tmp_root, Sum)))
    print('cs ', np.unique(children_sizes, return_counts=True))
    params = sum(children_sizes)
    params += 2 * len(get_nodes_by_type(tmp_root, Leaf))

    print("params", params)
    0/0


    train_im = np.random.randn(9 * 1000).reshape((1000, -1))

    add_parametric_inference_support()


    for spn in spns:
        valid, err = is_valid(spn)
        print(valid, err)
        print(get_structure_stats(spn))



        0 / 0
