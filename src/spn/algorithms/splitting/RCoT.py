import numpy as np
import os; path = os.path.dirname(__file__)
from tempfile import mkdtemp, TemporaryDirectory
from networkx.algorithms.components.connected import connected_components
from networkx.convert_matrix import from_numpy_matrix
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import multiprocessing as mp
from spn.algorithms.splitting.Base import split_conditional_data_by_clusters, preproc

with open(path+"/RCoT.R", "r") as mixfile:
   code = ''.join(mixfile.readlines())
   CoTest = SignatureTranslatedAnonymousPackage(code, "RCoT")

temp_path = TemporaryDirectory()
data_file = temp_path.name + '/data_file.dat'
data_cond_file = temp_path.name + '/data_cond_file.dat'

def remove_tmp_folder():
    temp_path.cleanup()

import atexit
atexit.register(remove_tmp_folder)


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

        output_mask = np.zeros(data.shape, dtype=bool)   # todo check scope and node.scope again
        output_mask[:, np.arange(len(scope))] = True


        dataOut = data[output_mask].reshape(num_instance, -1)
        dataIn = data[~output_mask].reshape(num_instance, -1)

        assert len(dataIn) > 0
        assert len(dataOut) > 0

        DATA = np.memmap(data_file, dtype=dataOut.dtype, mode='w+', shape=dataOut.shape)
        DATA[:] = dataOut[:]
        # DATA.flush()

        DATA_COND = np.memmap(data_cond_file, dtype=dataIn.dtype, mode='w+', shape=dataIn.shape)
        DATA_COND[:] = dataIn[:]
        DATA_COND.flush()

        num_X = dataOut.shape[1] #len(data[0])
        num_Y = dataIn.shape[1] #len(data_cond[0])
        p_value_matrix = np.zeros((num_X, num_X))


        index_matrix = [(x, y, dataOut.dtype, dataOut.shape, dataIn.shape) for x in range(num_X) for y in range(num_X)]
        index_matrix = [s for s in index_matrix if s[0]!=s[1]]

        with mp.Pool() as pool:
            pvals = pool.starmap(computePvals, index_matrix) #index_matrix)

        pvals_container = np.zeros((num_X,num_X))
        d = np.diag_indices_from(pvals_container)
        pvals_container[d] = 0

        pvals = np.array(pvals).ravel()
        for i in range(len(pvals)):
            x, y, _, _, _ = index_matrix[i]
            pvals_container[x,y] = pvals[i]

        pvals = pvals_container

        for i, j in zip(*np.tril_indices(pvals.shape[1])):
            pvals[i, j] = pvals[j, i] = min(pvals[i, j], pvals[j, i])

        pvals[pvals > alpha] = 0

        clusters = np.zeros(dataOut.shape[1])
        for i, c in enumerate(connected_components(from_numpy_matrix(pvals))):
            clusters[list(c)] = i + 1


        return split_conditional_data_by_clusters(local_data, clusters, scope, rows=False)

    return getCIGroups

def computePvals(x, y, data_type, data_shape, data_cond_shape):
    DATA = np.memmap(data_file, dtype=data_type, mode='r+', shape=data_shape)
    DATA_COND = np.memmap(data_cond_file, dtype=data_type, mode='r+', shape=data_cond_shape)

    num_inst = np.shape(DATA[:,x])[0]
    X_1 = np.asarray(DATA[:,x]).reshape(num_inst, 1)
    X_2 = np.asarray(DATA[:,y]).reshape(num_inst, 1)
    Y = np.asarray(DATA_COND)
    try:
        result = RcoT(X_1, X_2, Y)
        result = result[0][0]
    except Exception as e:
        print(e)
        np.savetxt('X', X_1, fmt='%d')
        np.savetxt('Y', X_2, fmt='%d')
        np.savetxt('Z', Y, fmt='%d')
        raise e

    return result


def RcoT(x,y,z):
    numpy2ri.activate()
    try:
        df_x = robjects.r["as.data.frame"](x)
        df_y = robjects.r["as.data.frame"](y)
        df_z = robjects.r["as.data.frame"](z)
        result = CoTest.RCoT(df_x,df_y,df_z)
        result = np.asarray(result)
    except Exception as e:
        print(e)
        print(z)
        print(z.shape)
        np.savetxt('X', x, fmt='%d')
        np.savetxt('Y', y, fmt='%d')
        np.savetxt('Z', z, fmt='%d')
        raise e

    return result
