'''
Created on March 28, 2018

@author: Alejandro Molina
'''
import numpy as np
from joblib import Memory

from spn.algorithms.Inference import likelihood, log_likelihood
from spn.algorithms.StructureLearning import learn_structure
from spn.algorithms.splitting.Clustering import get_split_rows_KMeans
from spn.algorithms.splitting.RDC import get_split_cols_RDC
from spn.gpu.TensorFlow import eval_tf, spn_to_tf_graph
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.histogram.Inference import add_histogram_inference_support

memory = Memory(cachedir="cache", verbose=0, compress=9)


#@memory.cache
def learn(data, ds_context):
    spn = learn_structure(data, ds_context, get_split_rows_KMeans(), get_split_cols_RDC(), create_histogram_leaf)

    return spn


if __name__ == '__main__':
    add_histogram_inference_support()
    np.random.seed(17)
    data = np.random.normal(10, 0.01, size=2000).tolist() + np.random.normal(30, 10, size=2000).tolist()
    data = np.array(data).reshape((-1, 10))
    data[data < 0] = 0
    data = (data * 1).astype(int)

    ds_context = Context(meta_types=[MetaType.DISCRETE] * data.shape[1])
    ds_context.add_domains(data)

    data[:,0] = 0
    data[:,1] = 1

    spn = learn(data, ds_context)
    spn = create_histogram_leaf(data[:, 0].reshape((-1, 1)), ds_context, [0], alpha=False, hist_source="kde") * \
          create_histogram_leaf(data[:, 1].reshape((-1, 1)), ds_context, [1], alpha=False, hist_source="kde")

    spn = 0.3 * create_histogram_leaf(data[:, 0].reshape((-1, 1)), ds_context, [0], alpha=False, hist_source="kde") + \
          0.7 * create_histogram_leaf(data[:, 0].reshape((-1, 1)), ds_context, [0], alpha=False, hist_source="kde")

    py_ll = log_likelihood(spn, data)

    tf_graph, placeholder = spn_to_tf_graph(spn, data)

    log_tf_out = eval_tf(tf_graph, placeholder, data)

    print("results are similar for Log TF and Python?", np.all(np.isclose(py_ll, log_tf_out)))
