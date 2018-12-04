from joblib import Memory
from matplotlib.colors import LogNorm, PowerNorm
from spn.algorithms.Inference import log_likelihood, likelihood
from spn.algorithms.LearningWrappers import learn_mspn, learn_parametric
from numpy import genfromtxt
import numpy as np
from spn.algorithms.Marginalization import marginalize
from spn.structure.Base import Context, Sum
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.histogram.Inference import add_histogram_inference_support
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical
from spn.structure.leaves.piecewise.Inference import add_piecewise_inference_support
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf
import matplotlib.cm as cm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


def create_leaf(data, ds_context, scope):
    return create_piecewise_leaf(data, ds_context, scope, isotonic=False, prior_weight=None)
    # return create_histogram_leaf(data, ds_context, scope, alpha=0.1)


add_piecewise_inference_support()
add_histogram_inference_support()
add_parametric_inference_support()
memory = Memory(cachedir="cache", verbose=0, compress=9)


data = []
for x in range(10):
    for y in range(10):
        for z in range(10):
            data.append([x, y, z, int(((x + y + z) / 5))])
data = np.array(data).astype(np.float)
types = [MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE]

ds_context = Context(meta_types=types)
ds_context.parametric_types = [Gaussian, Gaussian, Gaussian, Categorical]
ds_context.add_domains(data)

num_classes = len(np.unique(data[:, 3]))

# spn = learn_mspn(data, ds_context, min_instances_slice=10, leaves=create_leaf, threshold=0.3)

spn = Sum()
for label, count in zip(*np.unique(data[:, 3], return_counts=True)):
    branch = learn_mspn(
        data[data[:, 3] == label, :], ds_context, min_instances_slice=10, leaves=create_leaf, threshold=0.1
    )
    spn.children.append(branch)
    spn.weights.append(count / data.shape[0])

spn.scope.extend(branch.scope)


print("learned")

prediction = []

cls_data = np.zeros((num_classes, 4))
cls_data[:, 3] = np.arange(num_classes)


for i, x in enumerate(data):
    cls_data[:, 0:3] = x[0:3]
    prob = log_likelihood(spn, cls_data)
    prediction.append(np.argmax(prob))
    print(i)

print("Classification report:")

print(classification_report(data[:, 3], prediction))

print("Confusion matrix:")

print(confusion_matrix(data[:, 3], prediction))
