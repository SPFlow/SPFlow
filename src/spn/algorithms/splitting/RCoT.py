from time import perf_counter

import numpy as np
import os;

from tqdm import tqdm

path = os.path.dirname(__file__)
from tempfile import mkdtemp, TemporaryDirectory
from networkx.algorithms.components.connected import connected_components
from networkx.convert_matrix import from_numpy_matrix
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import multiprocessing as mp
from spn.algorithms.splitting.Base import split_conditional_data_by_clusters, preproc

with open(path + "/RCoT.R", "r") as mixfile:
    code = ''.join(mixfile.readlines())
    CoTest = SignatureTranslatedAnonymousPackage(code, "RCoT")


def getCIGroup(rand_gen, ohe=False):
    # def getCIGroups(data, scope=None, alpha=0.001, families=None):
    def getCIGroups(local_data, ds_context=None, scope=None, alpha=0.001, families=None):
        """
        :param local_data: np array
        :param scope: a list of index to output variables
        :param alpha: threshold
        :param families: obsolete
        :return: np array of clustering

        This function take tuple (output, conditional) as input and returns independent groups
        alpha is the cutoff parameter for connected components
        BE CAREFUL WITH SPARSE DATA!
        """

        data = preproc(local_data, ds_context, None, ohe)

        num_instance = data.shape[0]

        output_mask = np.zeros(data.shape, dtype=bool)  # todo check scope and node.scope again
        output_mask[:, np.arange(len(scope))] = True

        dataOut = data[output_mask].reshape(num_instance, -1)
        dataIn = data[~output_mask].reshape(num_instance, -1)

        assert dataIn.shape[0] > 0
        assert dataOut.shape[0] > 0
        assert dataOut.shape[1] > 1

        pvals = testRcoT(dataOut, dataIn)

        pvals[pvals > alpha] = 0

        clusters = np.zeros(dataOut.shape[1])
        for i, c in enumerate(connected_components(from_numpy_matrix(pvals))):
            clusters[list(c)] = i + 1

        return split_conditional_data_by_clusters(local_data, clusters, scope, rows=False)

    return getCIGroups


def testRcoT(DataOut, DataIn):
    numpy2ri.activate()
    try:
        df_DataIn = robjects.r["as.data.frame"](DataIn)
        df_DataOut = robjects.r["as.data.frame"](DataOut)
        result = CoTest.testRCoT(df_DataOut, df_DataIn)
        result = np.asarray(result)
    except Exception as e:
        np.savetxt('/tmp/dataIn.txt', DataIn)
        np.savetxt('/tmp/dataOut.txt', DataOut)
        raise e

    return result
