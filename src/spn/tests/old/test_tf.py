'''
Created on March 28, 2018

@author: Alejandro Molina
'''
import numpy as np
from joblib import Memory

from spn.algorithms.Inference import likelihood, histogram_likelihood
from spn.algorithms.StructureLearning import learn_structure
from spn.algorithms.splitting.Clustering import get_split_rows_KMeans
from spn.algorithms.splitting.RDC import get_split_cols_RDC
from spn.gpu.TensorFlow import eval_tf
from spn.structure.Base import Context
from spn.structure.leaves.Histograms import create_histogram_leaf, add_domains

from spn.structure.StatisticalTypes import MetaType

memory = Memory(cachedir="cache", verbose=0, compress=9)


@memory.cache
def learn(data, ds_context):
    spn = learn_structure(data, ds_context, get_split_rows_KMeans(), get_split_cols_RDC(), create_histogram_leaf)

    return spn


if __name__ == '__main__':
    np.random.seed(17)
    data = np.random.normal(10, 0.01, size=2000).tolist() + np.random.normal(30, 10, size=2000).tolist()
    data = np.array(data).reshape((-1, 10))

    ds_context = Context(meta_types=[MetaType.REAL] * data.shape[1])
    ds_context.add_domains(data)


    spn = learn(data, ds_context)
    py_ll = likelihood(spn, data, histogram_likelihood)

    tf_out, time = eval_tf(spn, data, log_space=False, save_graph_path='tfgraph', trace=True)
    tf_out, time = eval_tf(spn, data, log_space=False, save_graph_path='tfgraph', trace=True)
    tf_out_log, time2 = eval_tf(spn, data, log_space=True, save_graph_path='tfgraph')

    print("results are similar for TF and Python?", np.all(np.isclose(py_ll - np.log(tf_out), 0)))
    print("time in ns", time)
    print("results are similar for Log TF and Python?", np.all(np.isclose(py_ll - tf_out_log, 0)))
