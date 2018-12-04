"""
Created on September 04, 2018

@author: Alejandro Molina
"""
import logging

from joblib import Memory
from sklearn.metrics import hamming_loss, zero_one_loss, precision_score

from spn.algorithms.LearningWrappers import learn_conditional
from spn.algorithms.MPE import mpe
from spn.data.datasets import get_categorical_data
from spn.structure.Base import Context
from spn.structure.leaves.conditional.Conditional import Conditional_Gaussian, Conditional_Bernoulli
from spn.structure.leaves.conditional.Inference import add_conditional_inference_support
from spn.structure.leaves.conditional.MPE import add_conditional_mpe_support
from spn.structure.leaves.conditional.Sampling import add_conditional_sampling_support
from spn.structure.leaves.parametric.Parametric import Gaussian
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logging.captureWarnings(True)

memory = Memory(location="/tmp/ml_classification", verbose=10, compress=9)

add_conditional_inference_support()
add_conditional_sampling_support()
add_conditional_mpe_support()

if __name__ == "__main__":
    train_input, train_labels, test_input, test_labels = get_categorical_data("yeast")

    print(train_input.shape)
    print(train_labels.shape)
    print(test_input.shape)
    print(test_labels.shape)

    num_labels = train_labels.shape[1]

    ds_context = Context(parametric_types=[Conditional_Bernoulli] * num_labels)
    ds_context.add_domains(train_labels)

    train_data = np.concatenate((train_labels, train_input), axis=1)

    cspn = learn_conditional(
        train_data,
        ds_context,
        scope=list(range(num_labels)),
        rows="tsne",
        min_instances_slice=500,
        threshold=0.5,
        memory=memory,
    )

    test_data = np.zeros_like(test_labels, dtype=np.float32)
    test_data[:] = np.nan
    test_data = np.concatenate((test_data, test_input), axis=1)
    pred_test_labels = mpe(cspn, test_data)[:, 0:num_labels]

    # compare with
    # https://papers.nips.cc/paper/1964-a-kernel-method-for-multi-labelled-classification.pdf
    binary_pred_labels = np.round(pred_test_labels).astype(int)
    binary_pred_labels[binary_pred_labels < 0] = 0
    print("hamming_loss", hamming_loss(test_labels, binary_pred_labels))
    print("zero_one_loss", zero_one_loss(test_labels, binary_pred_labels))
    print("precision_score", precision_score(test_labels, binary_pred_labels, average="micro"))
