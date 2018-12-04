from joblib import Memory
from spn.algorithms.Inference import log_likelihood, likelihood
from spn.algorithms.LearningWrappers import learn_mspn, learn_parametric
import numpy as np
from spn.structure.Base import Context, Sum
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.histogram.Inference import add_histogram_inference_support
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.piecewise.Inference import add_piecewise_inference_support
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


def create_leaf(data, ds_context, scope):
    # return create_piecewise_leaf(data, ds_context, scope, isotonic=False, prior_weight=0.01)
    return create_histogram_leaf(data, ds_context, scope, alpha=0.005)


add_piecewise_inference_support()
add_histogram_inference_support()
add_parametric_inference_support()
memory = Memory(cachedir="cache", verbose=0, compress=9)

mnist = fetch_mldata("MNIST original")
X = mnist.data.astype("float64")
y = mnist.target
random_state = check_random_state(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10000, test_size=300, random_state=random_state)

# normalize
# scaler = StandardScaler()
# Xtrain = scaler.fit_transform(X_train)
# Xtest = scaler.transform(X_test)

data = []
i = 0
while i < len(X_train):
    data.append(np.append(X_train[i], y_train[i]))
    i += 1
data = np.array(data)

types = [MetaType.DISCRETE for i in range(784)]
types.append(MetaType.DISCRETE)

ds_context = Context(meta_types=types)
ds_context.add_domains(data)

spn = Sum()
for label, count in zip(*np.unique(data[:, 784], return_counts=True)):
    branch = learn_mspn(
        data[data[:, 784] == label, :], ds_context, min_instances_slice=2000, leaves=create_leaf, threshold=0.1
    )
    spn.children.append(branch)
    spn.weights.append(count / data.shape[0])
spn.scope.extend(branch.scope)

print("SPN learned")

num_classes = len(np.unique(data[:, 784]))
prediction = []
cls_data = np.zeros((num_classes, 785))
cls_data[:, 784] = np.arange(num_classes)

for i, x in enumerate(X_test):
    cls_data[:, 0:783] = x[0:783]
    prob = log_likelihood(spn, cls_data)
    prediction.append(np.argmax(prob))
    print(i)

print("predicted")

print("Classification accuracy:")

print(accuracy_score(y_test, prediction))

print("Confusion matrix:")

print(confusion_matrix(y_test, prediction))
