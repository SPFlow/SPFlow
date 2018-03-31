'''
Created on March 30, 2018

@author: Alejandro Molina
'''
from spn.algorithms.StructureLearning import get_next_operation, learn_structure
from spn.algorithms.splitting.Clustering import get_split_rows_KMeans
from spn.algorithms.splitting.RDC import get_split_cols_RDC
from spn.structure.leaves.Histograms import create_histogram_leaf


def learn_mspn(data, ds_context, cols="rdc", rows="kmeans", min_instances_slice=200, threshold=0.3, linear=False, ohe=False,
               memory=None):

    def learn(data, ds_context, cols, rows, min_instances_slice, threshold, linear, ohe):
        split_cols = None
        if cols == "rdc":
            split_cols = get_split_cols_RDC(threshold, ohe, linear)
        if rows == "kmeans":
            splot_rows = get_split_rows_KMeans()

        leaves = create_histogram_leaf

        nextop = get_next_operation(min_instances_slice)

        return learn_structure(data, ds_context, splot_rows, split_cols, leaves, nextop)

    if memory:
        learn = memory.cache(learn)

    return learn(data, ds_context, cols, rows, min_instances_slice, threshold, linear, ohe)