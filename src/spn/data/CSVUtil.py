"""
created 18/11/01
@author Claas Völcker
"""
from spn.algorithms.LeafLearning import learn_leaf_from_context
from spn.algorithms.LearningWrappers import learn_mspn
from spn.algorithms.StructureLearning import is_valid

from spn.data.datasets import load_from_csv

from spn.structure.Base import Context
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear

def learn_piecewise_from_file(data_file, header=0, min_instances=25, independence_threshold=0.1):
    """
    Learning wrapper for automatically building an SPN from a datafile

    :param data_file: String: location of the data csv
    :param header: Int: row of the data header
    :param min_instances: Int: minimum data instances per leaf node
    :param independence_threshold: Float: threshold for the independence test
    :param histogram: Boolean: use histogram for categorical data?
    :return: a valid spn, a data dictionary
    """
    data, feature_types, data_dictionary = load_from_csv(data_file, header, histogram)
    feature_classes = [Histogram if name == 'hist' else PiecewiseLinear for name in feature_types]
    context = Context(parametric_types=feature_classes).add_domains(data)
    context.add_feature_names([entry['name']
                                  for entry in data_dictionary['features']])
    spn = learn_mspn(data,
                     context,
                     min_instances_slice=min_instances,
                     threshold=independence_threshold,
                     ohe=False,
                     leaves=create_piecewise_leaf)
    assert is_valid(spn), 'No valid spn could be created from datafile'
    data_dictionary['context'] = context
    return spn, data_dictionary
