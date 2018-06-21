'''
Created on June 20, 2018

@author: Alejandro Molina
'''
from spn.experiments.RandomSPNs.LearnRGSPN import Make_SPN_from_RegionGraph
from spn.experiments.RandomSPNs.region_graph import RegionGraph
import numpy as np

from spn.io.CPP import to_cpp

rg = RegionGraph(range(28 * 28))
for _ in range(0, 2):
    # for _ in range(0, 20):
    rg.random_split(2, 2)

rg_layers = rg.make_layers()

num_classes = 10

vector_list, tmp_root = Make_SPN_from_RegionGraph(rg_layers, np.random.RandomState(100),
                                        num_classes=num_classes, num_gauss=5, num_sums=5)


cpp_code = to_cpp(tmp_root, c_data_type="double")

print(tmp_root)

text_file = open("spn.c", "w")
text_file.write(cpp_code)
text_file.close()