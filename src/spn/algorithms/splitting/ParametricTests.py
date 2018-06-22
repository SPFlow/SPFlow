'''
Created on June 21, 2018

@author: Alejandro Molina
'''

import numpy as np

def get_split_cols_GTest(threshold=0.3):
    def split_cols_RDC(local_data, ds_context, scope):
        adjm = np.zero((local_data.shape[1], local_data.shape[1]))

        


        clusters = clusters_by_adjacency_matrix(adjm, threshold, local_data.shape[1])

        return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_RDC