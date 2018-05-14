'''
Created on March 20, 2018

@author: Alejandro Molina
'''

from spn.algorithms import Inference
from spn.algorithms.StructureLearning import learn_structure
from spn.algorithms.splitting.Clustering import get_split_rows_KMeans
from spn.algorithms.splitting.RDC import get_split_cols_RDC
from spn.data.datasets import get_nips_data
from spn.structure.Base import Context
from spn.structure.leaves.Histograms import add_domains, create_histogram_leaf

if __name__ == '__main__':
    import numpy as np

    ds_name, words, data, train, _, statistical_type, _ = get_nips_data()

    print(words)

    print(data)

    ds_context = Context()
    ds_context.statistical_type = np.asarray(["discrete"] * data.shape[1])

    add_domains(data, ds_context)

    spn = learn_structure(data, ds_context, get_split_rows_KMeans(), get_split_cols_RDC(), create_histogram_leaf)

    # print(to_str_equation(spn, words))
    print(Inference.likelihood(spn, data[0:100, :]))
